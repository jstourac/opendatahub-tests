import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from pyhelper_utils.shell import run_command

from tests.workbenches.notebooks_server.controller.utils import build_notebook_dict, get_username
from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)

NOTEBOOK_PORT = 8888


@pytest.fixture(scope="class")
def second_notebook_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    users_persistent_volume_claim: PersistentVolumeClaim,
    notebook_image: str,
) -> Pod:
    """Create a second notebook in the same namespace for network policy testing."""
    namespace = unprivileged_model_namespace.name
    name = "test-nb-netpol-second"

    username = get_username(client=admin_client)
    assert username, "Failed to determine username from the cluster"

    notebook_dict = build_notebook_dict(
        namespace=namespace,
        name=name,
        image_path=notebook_image,
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict):
        pod = Pod(
            client=unprivileged_client,
            namespace=namespace,
            name=f"{name}-0",
        )
        pod.wait(timeout=Timeout.TIMEOUT_1MIN)
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_2MIN,
        )
        yield pod


class TestNetworkPolicyIsolation:
    """Verify network policy isolation between notebook pods."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-netpol",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-netpol"},
                {
                    "namespace": "test-nb-netpol",
                    "name": "test-nb-netpol-first",
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_notebooks_cannot_reach_each_other",
            )
        ],
        indirect=True,
    )
    def test_notebooks_cannot_reach_each_other(
        self,
        admin_client: DynamicClient,
        notebook_pod: Pod,
        second_notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify that two notebook pods in the same namespace cannot reach each other.

        Given two notebook pods are running in the same namespace,
        When pod B tries to connect to pod A on the notebook port,
        Then the connection should be refused or time out due to NetworkPolicy.
        """
        first_pod_ip = notebook_pod.instance.status.podIP
        assert first_pod_ip, f"Could not determine IP for pod {notebook_pod.name}"

        second_pod_name = second_notebook_pod.name
        namespace = second_notebook_pod.namespace

        curl_cmd = (
            f"oc exec {second_pod_name} -n {namespace} -- "
            f"curl -s -o /dev/null -w '%{{http_code}}' "
            f"--connect-timeout 5 --max-time 10 "
            f"http://{first_pod_ip}:{NOTEBOOK_PORT}/api"
        )

        exit_code, stdout, _ = run_command(
            command=curl_cmd.split(),
            verify_stderr=False,
            check=False,
        )

        connection_blocked = exit_code != 0 or stdout.strip() not in ("200", "302", "403")

        assert connection_blocked, (
            f"Pod '{second_pod_name}' was able to reach pod '{notebook_pod.name}' "
            f"at {first_pod_ip}:{NOTEBOOK_PORT} (HTTP {stdout.strip()}). "
            f"NetworkPolicy should prevent this connection."
        )
