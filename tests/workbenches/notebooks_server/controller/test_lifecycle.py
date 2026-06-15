import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.workbenches.notebooks_server.controller.utils import build_notebook_dict, get_username
from utilities.constants import Timeout

LOGGER = structlog.get_logger(name=__name__)

KUBEFLOW_STOPPED_ANNOTATION = "kubeflow-resource-stopped"


def _wait_for_pod_absence(pod: Pod, timeout: int) -> bool:
    """Wait until the pod no longer exists."""
    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: not pod.exists,
        ):
            if sample:
                return True
    except TimeoutExpiredError:
        return False
    return False


class TestNotebookLifecycle:
    """Verify notebook CR delete/recreate and stop/start lifecycle operations."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-delete-recreate",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-delete-recreate"},
                {
                    "namespace": "test-nb-delete-recreate",
                    "name": "test-nb-delete-recreate",
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_delete_and_recreate_notebook",
            )
        ],
        indirect=True,
    )
    def test_delete_and_recreate_notebook(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        notebook_image: str,
    ) -> None:
        """Verify that deleting a Notebook CR removes the pod and a new CR can be created.

        Given a running notebook pod,
        When the Notebook CR is deleted,
        Then the pod should be removed.
        When a new Notebook CR is created with the same name,
        Then a new pod should reach Ready state.
        """
        notebook_name = default_notebook.name
        namespace = default_notebook.namespace

        assert notebook_pod.exists, "Notebook pod should exist before deletion"

        default_notebook.delete(wait=True, timeout=Timeout.TIMEOUT_2MIN)

        pod_removed = _wait_for_pod_absence(pod=notebook_pod, timeout=Timeout.TIMEOUT_2MIN)
        assert pod_removed, (
            f"Pod '{notebook_pod.name}' was not removed after Notebook CR deletion within {Timeout.TIMEOUT_2MIN}s"
        )

        username = get_username(client=admin_client)
        assert username, "Failed to determine username from the cluster"

        notebook_dict = build_notebook_dict(
            namespace=namespace,
            name=notebook_name,
            image_path=notebook_image,
        )

        with Notebook(client=unprivileged_client, kind_dict=notebook_dict):
            new_pod = Pod(
                client=unprivileged_client,
                namespace=namespace,
                name=f"{notebook_name}-0",
            )
            new_pod.wait(timeout=Timeout.TIMEOUT_1MIN)
            new_pod.wait_for_condition(
                condition=Pod.Condition.READY,
                status=Pod.Condition.Status.TRUE,
                timeout=Timeout.TIMEOUT_2MIN,
            )

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-stop-start",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-stop-start"},
                {
                    "namespace": "test-nb-stop-start",
                    "name": "test-nb-stop-start",
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_stop_and_start_notebook",
            )
        ],
        indirect=True,
    )
    def test_stop_and_start_notebook(
        self,
        unprivileged_client: DynamicClient,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify stop (annotation) removes the pod and removing the annotation restarts it.

        Given a running notebook pod,
        When the kubeflow-resource-stopped annotation is added,
        Then the pod should be removed (StatefulSet scaled to 0).
        When the annotation is removed,
        Then the pod should be recreated and reach Ready state.
        """
        notebook_name = default_notebook.name
        namespace = default_notebook.namespace

        assert notebook_pod.exists, "Notebook pod should exist before stop"

        default_notebook.instance.metadata.annotations[KUBEFLOW_STOPPED_ANNOTATION] = "true"
        default_notebook.update(resource_dict=default_notebook.instance.to_dict())

        pod_removed = _wait_for_pod_absence(pod=notebook_pod, timeout=Timeout.TIMEOUT_2MIN)
        assert pod_removed, (
            f"Pod '{notebook_pod.name}' was not removed after adding stop annotation within {Timeout.TIMEOUT_2MIN}s"
        )

        annotations = dict(default_notebook.instance.metadata.annotations)
        annotations.pop(KUBEFLOW_STOPPED_ANNOTATION, None)
        default_notebook.instance.metadata.annotations = annotations
        default_notebook.update(resource_dict=default_notebook.instance.to_dict())

        restarted_pod = Pod(
            client=unprivileged_client,
            namespace=namespace,
            name=f"{notebook_name}-0",
        )
        restarted_pod.wait(timeout=Timeout.TIMEOUT_1MIN)
        restarted_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_2MIN,
        )
