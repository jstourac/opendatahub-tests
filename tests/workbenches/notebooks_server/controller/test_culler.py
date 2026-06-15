from collections.abc import Generator

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)

CULLER_CONFIGMAP_NAME = "notebook-controller-culler-config"
CULLER_TIMEOUT_KEY = "CULL_IDLE_TIME"
CULLER_ENABLE_KEY = "ENABLE_CULLING"
CONTROLLER_LABEL_SELECTOR = "app=notebook-controller"


@pytest.fixture(scope="class")
def culler_configmap(admin_client: DynamicClient) -> ConfigMap | None:
    """Retrieve the notebook-controller-culler-config ConfigMap if it exists."""
    namespace = py_config["applications_namespace"]
    cm = ConfigMap(
        client=admin_client,
        name=CULLER_CONFIGMAP_NAME,
        namespace=namespace,
    )
    if cm.exists:
        return cm
    return None


@pytest.fixture(scope="class")
def controller_pod(admin_client: DynamicClient) -> Pod:
    """Retrieve the notebook-controller (KF) pod."""
    namespace = py_config["applications_namespace"]
    pods = list(
        Pod.get(
            client=admin_client,
            namespace=namespace,
            label_selector=CONTROLLER_LABEL_SELECTOR,
        )
    )
    assert pods, f"No notebook-controller pods found in {namespace}"
    return pods[0]


@pytest.fixture(scope="function")
def short_culler_timeout(admin_client: DynamicClient, culler_configmap: ConfigMap | None) -> Generator[ConfigMap]:
    """Temporarily set a short culler timeout (60s) for testing culler behavior.

    Restores the original value (or removes the ConfigMap) on teardown.
    """
    namespace = py_config["applications_namespace"]
    short_timeout = "1"  # 1 minute

    if culler_configmap and culler_configmap.exists:
        with ResourceEditor(
            patches={
                culler_configmap: {
                    "data": {
                        CULLER_TIMEOUT_KEY: short_timeout,
                        CULLER_ENABLE_KEY: "true",
                    }
                }
            }
        ):
            yield culler_configmap
    else:
        with ConfigMap(
            client=admin_client,
            name=CULLER_CONFIGMAP_NAME,
            namespace=namespace,
            data={
                CULLER_TIMEOUT_KEY: short_timeout,
                CULLER_ENABLE_KEY: "true",
            },
        ) as cm:
            yield cm


def _wait_for_pod_deletion(pod: Pod, timeout: int) -> bool:
    """Wait until the pod no longer exists."""
    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=10,
            func=lambda: not pod.exists,
        ):
            if sample:
                return True
    except TimeoutExpiredError:
        return False
    return False


class TestCullerConfiguration:
    """Verify notebook-controller culler configuration is consistent."""

    @pytest.mark.smoke
    def test_culler_configmap_structure(
        self,
        culler_configmap: ConfigMap | None,
    ) -> None:
        """Verify the culler ConfigMap exists and contains expected keys.

        Given the notebook culler is enabled on the cluster,
        When reading the culler ConfigMap,
        Then it should contain CULL_IDLE_TIME and ENABLE_CULLING fields.
        """
        if culler_configmap is None:
            pytest.skip(f"ConfigMap '{CULLER_CONFIGMAP_NAME}' not found - culler may be disabled")

        cm_data = culler_configmap.instance.data
        assert cm_data, f"ConfigMap '{CULLER_CONFIGMAP_NAME}' has no data"

        assert CULLER_TIMEOUT_KEY in cm_data, (
            f"ConfigMap missing key '{CULLER_TIMEOUT_KEY}'. Keys present: {list(cm_data.keys())}"
        )
        assert CULLER_ENABLE_KEY in cm_data, (
            f"ConfigMap missing key '{CULLER_ENABLE_KEY}'. Keys present: {list(cm_data.keys())}"
        )

        timeout_value = cm_data[CULLER_TIMEOUT_KEY]
        assert timeout_value.isdigit(), (
            f"'{CULLER_TIMEOUT_KEY}' value should be numeric (minutes), got: '{timeout_value}'"
        )

    @pytest.mark.smoke
    def test_controller_pod_env_matches_configmap(
        self,
        culler_configmap: ConfigMap | None,
        controller_pod: Pod,
    ) -> None:
        """Verify the controller pod CULL_IDLE_TIME env matches the ConfigMap value.

        Given the culler ConfigMap is set with a specific timeout,
        When inspecting the controller pod environment,
        Then the CULL_IDLE_TIME env variable should match the ConfigMap data.
        """
        if culler_configmap is None:
            pytest.skip(f"ConfigMap '{CULLER_CONFIGMAP_NAME}' not found - culler may be disabled")

        cm_timeout = culler_configmap.instance.data.get(CULLER_TIMEOUT_KEY)

        manager_container = None
        for container in controller_pod.instance.spec.containers:
            if container.name == "manager":
                manager_container = container
                break

        assert manager_container, f"Container 'manager' not found in pod {controller_pod.name}"

        pod_env_timeout = None
        if manager_container.env:
            for env_var in manager_container.env:
                if env_var.name == CULLER_TIMEOUT_KEY:
                    pod_env_timeout = env_var.value
                    break

        if manager_container.envFrom:
            for env_from in manager_container.envFrom:
                if env_from.configMapRef and env_from.configMapRef.name == CULLER_CONFIGMAP_NAME:
                    pod_env_timeout = cm_timeout
                    break

        assert pod_env_timeout is not None, (
            f"Environment variable '{CULLER_TIMEOUT_KEY}' not found in controller pod "
            f"'{controller_pod.name}' manager container"
        )
        assert pod_env_timeout == cm_timeout, (
            f"Controller pod env '{CULLER_TIMEOUT_KEY}={pod_env_timeout}' does not match ConfigMap value '{cm_timeout}'"
        )


class TestCullerBehavior:
    """Verify notebook culler correctly stops idle notebooks."""

    @pytest.mark.tier2
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-culler-idle",
                    "add-dashboard-label": True,
                },
                {"name": "test-culler-idle"},
                {
                    "namespace": "test-culler-idle",
                    "name": "test-culler-idle",
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_culler_stops_idle_notebook",
            )
        ],
        indirect=True,
    )
    def test_culler_stops_idle_notebook(
        self,
        short_culler_timeout: ConfigMap,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify that the culler stops an idle notebook after the timeout.

        Given a notebook is running with no kernel activity,
        When the culler timeout (1 minute) plus drift passes,
        Then the notebook pod should be deleted by the culler.
        """
        assert notebook_pod.exists, "Notebook pod should exist before culler acts"

        cull_wait = Timeout.TIMEOUT_4MIN
        pod_was_deleted = _wait_for_pod_deletion(pod=notebook_pod, timeout=cull_wait)

        assert pod_was_deleted, (
            f"Notebook pod '{notebook_pod.name}' was not culled within {cull_wait}s. "
            f"The culler may not be operating correctly."
        )
