import pytest
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod
from ocp_resources.service import Service

from tests.workbenches.notebooks_server.controller.utils import StatefulSet
from utilities.resources.http_route import HTTPRoute

KUBE_RBAC_PROXY_CONTAINER = "kube-rbac-proxy"


class TestPostUpgradeNotebookCreation:
    """Verify a new notebook can be created on the upgraded platform.

    Steps:
        1. Create a fresh Notebook CR on the upgraded controller.
        2. Verify the pod reaches Ready state.
        3. Verify the StatefulSet and Service are created.
        4. Verify the HTTPRoute is created for routing.
        5. Verify the kube-rbac-proxy sidecar is injected.
        6. Clean up the notebook.
    """

    @pytest.mark.post_upgrade
    def test_new_notebook_pod_ready(
        self,
        new_notebook_pod: Pod,
    ) -> None:
        """Given the platform was upgraded,
        When a new Notebook CR is created,
        Then the notebook pod should reach Ready state.
        """
        assert new_notebook_pod.exists, f"Pod '{new_notebook_pod.name}' was not created on upgraded platform"

    @pytest.mark.post_upgrade
    def test_new_notebook_statefulset_exists(
        self,
        new_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the controller reconciles,
        Then a StatefulSet should be created with 1 replica.
        """
        assert new_notebook_statefulset.exists, (
            f"StatefulSet '{new_notebook_statefulset.name}' was not created on upgraded platform"
        )

        replicas = new_notebook_statefulset.instance.spec.replicas
        assert replicas == 1, f"StatefulSet has {replicas} replicas, expected 1"

    @pytest.mark.post_upgrade
    def test_new_notebook_service_exists(
        self,
        new_notebook_service: Service,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the controller reconciles,
        Then a Service should be created.
        """
        assert new_notebook_service.exists, (
            f"Service '{new_notebook_service.name}' was not created on upgraded platform"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_httproute_exists(
        self,
        new_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the ODH controller reconciles routing,
        Then an HTTPRoute should be created.
        """
        assert new_notebook_httproute.exists, (
            f"HTTPRoute '{new_notebook_httproute.name}' was not created on upgraded platform"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_has_auth_sidecar(
        self,
        new_notebook_pod: Pod,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-auth,
        When the ODH webhook mutates the CR,
        Then the pod should have a kube-rbac-proxy sidecar container.
        """
        containers = new_notebook_pod.instance.spec.containers
        container_names = [c.name for c in containers]

        assert KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Pod '{new_notebook_pod.name}' missing '{KUBE_RBAC_PROXY_CONTAINER}' sidecar on upgraded platform. "
            f"Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_reconciliation_lock_cleared(
        self,
        new_notebook: Notebook,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the ODH controller completes its first reconciliation,
        Then the reconciliation lock annotation should be cleared.
        """
        stop_annotation = new_notebook.instance.metadata.annotations.get("kubeflow-resource-stopped")
        assert stop_annotation != "odh-notebook-controller-lock", (
            f"Notebook '{new_notebook.name}' still has reconciliation lock "
            f"'odh-notebook-controller-lock' after pod reached Ready"
        )
