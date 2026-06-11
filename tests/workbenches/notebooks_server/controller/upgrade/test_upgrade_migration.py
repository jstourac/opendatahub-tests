from typing import Any

import pytest
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from pytest_testconfig import config as py_config

from utilities.resources.http_route import HTTPRoute
from utilities.resources.reference_grant import ReferenceGrant

pytestmark = pytest.mark.order(index="last")

OAUTH_PROXY_CONTAINER = "oauth-proxy"
KUBE_RBAC_PROXY_CONTAINER = "kube-rbac-proxy"
INJECT_OAUTH_ANNOTATION = "notebooks.opendatahub.io/inject-oauth"
INJECT_AUTH_ANNOTATION = "notebooks.opendatahub.io/inject-auth"


class TestPostUpgrade2xResourcesSurvival:
    """Phase A: Verify 2.x workbench state after upgrade (before manual migration).

    The 3.x controller keeps old 2.x resources (Route, oauth-proxy, inject-oauth) intact
    but proactively creates HTTPRoutes for dual routing. Only runs for 2.x -> 3.x upgrades.
    """

    @pytest.fixture(autouse=True)
    def _skip_unless_2x_migration(self, is_migration_from_2x: bool) -> None:
        """Skip all tests in this class unless upgrading from 2.x."""
        if not is_migration_from_2x:
            pytest.skip("2.x resource survival checks only apply for 2.x -> 3.x upgrades")

    @pytest.mark.post_upgrade
    def test_stopped_notebook_route_still_exists(
        self,
        stopped_notebook_route: Route,
    ) -> None:
        """Given a stopped notebook was created on 2.x with an OpenShift Route,
        When the platform is upgraded to 3.x,
        Then the Route should still exist (migration not triggered until restart).
        """
        assert stopped_notebook_route.exists, (
            f"Route '{stopped_notebook_route.name}' no longer exists after upgrade for stopped notebook"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_has_inject_oauth_annotation(
        self,
        stopped_notebook: Notebook,
    ) -> None:
        """Given a stopped notebook was created on 2.x with inject-oauth,
        When the platform is upgraded to 3.x,
        Then the inject-oauth annotation should still be present.
        """
        annotations = stopped_notebook.instance.metadata.annotations or {}
        assert annotations.get(INJECT_OAUTH_ANNOTATION) == "true", (
            f"Notebook '{stopped_notebook.name}' lost '{INJECT_OAUTH_ANNOTATION}' annotation after upgrade. "
            f"Annotations: {list(annotations.keys())}"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_httproute_created_after_upgrade(
        self,
        admin_client: Any,
        stopped_notebook: Notebook,
    ) -> None:
        """Given a stopped notebook was created on 2.x,
        When the platform is upgraded to 3.x,
        Then the controller proactively creates an HTTPRoute (dual routing with the old Route).
        """
        apps_ns = py_config["applications_namespace"]
        httproute_name = f"nb-{stopped_notebook.namespace}-{stopped_notebook.name}"
        httproute = HTTPRoute(
            client=admin_client,
            name=httproute_name,
            namespace=apps_ns,
        )
        assert httproute.exists, (
            f"HTTPRoute '{httproute_name}' not found in '{apps_ns}' for stopped 2.x notebook after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_route_still_exists(
        self,
        upgrade_notebook_route: Route,
    ) -> None:
        """Given a running notebook was created on 2.x with an OpenShift Route,
        When the platform is upgraded to 3.x,
        Then the Route should still exist (migration not triggered until restart).
        """
        assert upgrade_notebook_route.exists, (
            f"Route '{upgrade_notebook_route.name}' no longer exists after upgrade for running notebook"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_has_inject_oauth_annotation(
        self,
        upgrade_notebook: Notebook,
    ) -> None:
        """Given a running notebook was created on 2.x with inject-oauth,
        When the platform is upgraded to 3.x,
        Then the inject-oauth annotation should still be present.
        """
        annotations = upgrade_notebook.instance.metadata.annotations or {}
        assert annotations.get(INJECT_OAUTH_ANNOTATION) == "true", (
            f"Notebook '{upgrade_notebook.name}' lost '{INJECT_OAUTH_ANNOTATION}' annotation after upgrade. "
            f"Annotations: {list(annotations.keys())}"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_oauth_proxy_sidecar_present(
        self,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Given a running notebook was created on 2.x with oauth-proxy sidecar,
        When the platform is upgraded to 3.x,
        Then the pod should still have the oauth-proxy container.
        """
        containers = upgrade_notebook_pod.instance.spec.containers
        container_names = [container.name for container in containers]

        assert OAUTH_PROXY_CONTAINER in container_names, (
            f"Pod '{upgrade_notebook_pod.name}' lost '{OAUTH_PROXY_CONTAINER}' sidecar after upgrade. "
            f"Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_not_restarted(
        self,
        upgrade_notebook_pod: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a running notebook existed before the 2.x-to-3.x upgrade,
        When the upgrade completes,
        Then the pod should not have been restarted (creationTimestamp unchanged).
        """
        current_timestamp = upgrade_notebook_pod.instance.metadata.creationTimestamp
        saved_timestamp = upgrade_notebook_baseline["ntb_creation_timestamp"]

        assert current_timestamp == saved_timestamp, (
            f"Running notebook pod was restarted during 2.x-to-3.x upgrade. "
            f"Pre-upgrade creationTimestamp: {saved_timestamp}, "
            f"post-upgrade creationTimestamp: {current_timestamp}"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_httproute_created_after_upgrade(
        self,
        admin_client: Any,
        upgrade_notebook: Notebook,
    ) -> None:
        """Given a running notebook was created on 2.x,
        When the platform is upgraded to 3.x,
        Then the controller proactively creates an HTTPRoute (dual routing with the old Route).
        """
        apps_ns = py_config["applications_namespace"]
        httproute_name = f"nb-{upgrade_notebook.namespace}-{upgrade_notebook.name}"
        httproute = HTTPRoute(
            client=admin_client,
            name=httproute_name,
            namespace=apps_ns,
        )
        assert httproute.exists, (
            f"HTTPRoute '{httproute_name}' not found in '{apps_ns}' for running 2.x notebook after upgrade"
        )


class TestPostUpgradeStoppedNotebookMigration:
    """Phase B: Verify stopped workbench migrates to 3.x when restarted after 2.x upgrade.

    Only runs when upgrading from 2.x. Skipped for 3.x-to-3.y upgrades.
    Must run AFTER Phase A tests (enforced by fixture dependency on restart_stopped_notebook).
    """

    @pytest.fixture(autouse=True)
    def _skip_unless_2x_migration(self, is_migration_from_2x: bool) -> None:
        """Skip all tests in this class unless upgrading from 2.x."""
        if not is_migration_from_2x:
            pytest.skip("Migration tests only apply for 2.x -> 3.x upgrades")

    @pytest.mark.post_upgrade
    def test_stopped_notebook_starts_after_restart(
        self,
        restart_stopped_notebook: Pod,
    ) -> None:
        """Given a stopped 2.x notebook is restarted after upgrade,
        When the stop annotation is removed,
        Then the notebook pod should reach Ready state.

        Validation is performed by the restart_stopped_notebook fixture which waits
        for the pod to exist and reach Ready condition.
        """

    @pytest.mark.post_upgrade
    def test_stopped_notebook_route_removed_after_restart(
        self,
        restart_stopped_notebook: Pod,
        stopped_notebook_route: Route,
    ) -> None:
        """Given a stopped 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then the old OpenShift Route should be removed.
        """
        assert not stopped_notebook_route.exists, (
            f"Route '{stopped_notebook_route.name}' still exists after migration restart"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_httproute_created_after_restart(
        self,
        admin_client: Any,
        restart_stopped_notebook: Pod,
        stopped_notebook: Notebook,
    ) -> None:
        """Given a stopped 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then an HTTPRoute should be created in the applications namespace.
        """
        apps_ns = py_config["applications_namespace"]
        httproute_name = f"nb-{stopped_notebook.namespace}-{stopped_notebook.name}"
        httproute = HTTPRoute(
            client=admin_client,
            name=httproute_name,
            namespace=apps_ns,
        )
        assert httproute.exists, f"HTTPRoute '{httproute_name}' not created in '{apps_ns}' after migration restart"

    @pytest.mark.post_upgrade
    def test_stopped_notebook_reference_grant_created_after_restart(
        self,
        admin_client: Any,
        restart_stopped_notebook: Pod,
        stopped_notebook: Notebook,
    ) -> None:
        """Given a stopped 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then a ReferenceGrant should be created in the notebook namespace.
        """
        ref_grant = ReferenceGrant(
            client=admin_client,
            name="notebook-httproute-access",
            namespace=stopped_notebook.namespace,
        )
        assert ref_grant.exists, (
            f"ReferenceGrant 'notebook-httproute-access' not created in "
            f"'{stopped_notebook.namespace}' after migration restart"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_inject_auth_annotation_after_restart(
        self,
        restart_stopped_notebook: Pod,
        stopped_notebook: Notebook,
    ) -> None:
        """Given a stopped 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then inject-auth should replace inject-oauth annotation.
        """
        annotations = stopped_notebook.instance.metadata.annotations or {}
        assert annotations.get(INJECT_AUTH_ANNOTATION) == "true", (
            f"Notebook '{stopped_notebook.name}' missing '{INJECT_AUTH_ANNOTATION}' after migration. "
            f"Annotations: {list(annotations.keys())}"
        )
        assert INJECT_OAUTH_ANNOTATION not in annotations, (
            f"Notebook '{stopped_notebook.name}' still has '{INJECT_OAUTH_ANNOTATION}' after migration"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_kube_rbac_proxy_sidecar_after_restart(
        self,
        restart_stopped_notebook: Pod,
    ) -> None:
        """Given a stopped 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then the pod should have kube-rbac-proxy sidecar instead of oauth-proxy.
        """
        containers = restart_stopped_notebook.instance.spec.containers
        container_names = [container.name for container in containers]

        assert KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Pod '{restart_stopped_notebook.name}' missing '{KUBE_RBAC_PROXY_CONTAINER}' "
            f"after migration. Containers: {container_names}"
        )
        assert OAUTH_PROXY_CONTAINER not in container_names, (
            f"Pod '{restart_stopped_notebook.name}' still has '{OAUTH_PROXY_CONTAINER}' "
            f"after migration. Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_main_image_unchanged_after_restart(
        self,
        restart_stopped_notebook: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a stopped 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then the main container image should remain unchanged.
        """
        containers = restart_stopped_notebook.instance.spec.containers
        main_container = next(
            (
                container
                for container in containers
                if container.name not in (KUBE_RBAC_PROXY_CONTAINER, OAUTH_PROXY_CONTAINER)
            ),
            None,
        )
        assert main_container is not None, (
            f"No main container found in pod '{restart_stopped_notebook.name}'. "
            f"Containers: {[container.name for container in containers]}"
        )

        saved_image = upgrade_notebook_baseline.get("notebook_image")
        if saved_image:
            assert main_container.image == saved_image, (
                f"Main container image changed after migration. Expected: {saved_image}, got: {main_container.image}"
            )


class TestPostUpgradeRunningNotebookMigration:
    """Phase B: Verify running workbench migrates to 3.x when restarted after 2.x upgrade.

    Only runs when upgrading from 2.x. Skipped for 3.x-to-3.y upgrades.
    Must run AFTER Phase A tests (enforced by fixture dependency on restart_running_notebook).
    """

    @pytest.fixture(autouse=True)
    def _skip_unless_2x_migration(self, is_migration_from_2x: bool) -> None:
        """Skip all tests in this class unless upgrading from 2.x."""
        if not is_migration_from_2x:
            pytest.skip("Migration tests only apply for 2.x -> 3.x upgrades")

    @pytest.mark.post_upgrade
    def test_running_notebook_restarts_after_migration(
        self,
        restart_running_notebook: Pod,
    ) -> None:
        """Given a running 2.x notebook is restarted after upgrade,
        When the stop-then-start cycle completes,
        Then the new notebook pod should reach Ready state.

        Validation is performed by the restart_running_notebook fixture which waits
        for the pod to exist and reach Ready condition.
        """

    @pytest.mark.post_upgrade
    def test_running_notebook_route_removed_after_restart(
        self,
        restart_running_notebook: Pod,
        upgrade_notebook_route: Route,
    ) -> None:
        """Given a running 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then the old OpenShift Route should be removed.
        """
        assert not upgrade_notebook_route.exists, (
            f"Route '{upgrade_notebook_route.name}' still exists after migration restart"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_httproute_created_after_restart(
        self,
        admin_client: Any,
        restart_running_notebook: Pod,
        upgrade_notebook: Notebook,
    ) -> None:
        """Given a running 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then an HTTPRoute should be created in the applications namespace.
        """
        apps_ns = py_config["applications_namespace"]
        httproute_name = f"nb-{upgrade_notebook.namespace}-{upgrade_notebook.name}"
        httproute = HTTPRoute(
            client=admin_client,
            name=httproute_name,
            namespace=apps_ns,
        )
        assert httproute.exists, f"HTTPRoute '{httproute_name}' not created in '{apps_ns}' after migration restart"

    @pytest.mark.post_upgrade
    def test_running_notebook_reference_grant_created_after_restart(
        self,
        admin_client: Any,
        restart_running_notebook: Pod,
        upgrade_notebook: Notebook,
    ) -> None:
        """Given a running 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then a ReferenceGrant should be created in the notebook namespace.
        """
        ref_grant = ReferenceGrant(
            client=admin_client,
            name="notebook-httproute-access",
            namespace=upgrade_notebook.namespace,
        )
        assert ref_grant.exists, (
            f"ReferenceGrant 'notebook-httproute-access' not created in "
            f"'{upgrade_notebook.namespace}' after migration restart"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_inject_auth_annotation_after_restart(
        self,
        restart_running_notebook: Pod,
        upgrade_notebook: Notebook,
    ) -> None:
        """Given a running 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then inject-auth should replace inject-oauth annotation.
        """
        annotations = upgrade_notebook.instance.metadata.annotations or {}
        assert annotations.get(INJECT_AUTH_ANNOTATION) == "true", (
            f"Notebook '{upgrade_notebook.name}' missing '{INJECT_AUTH_ANNOTATION}' after migration. "
            f"Annotations: {list(annotations.keys())}"
        )
        assert INJECT_OAUTH_ANNOTATION not in annotations, (
            f"Notebook '{upgrade_notebook.name}' still has '{INJECT_OAUTH_ANNOTATION}' after migration"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_kube_rbac_proxy_sidecar_after_restart(
        self,
        restart_running_notebook: Pod,
    ) -> None:
        """Given a running 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then the pod should have kube-rbac-proxy sidecar instead of oauth-proxy.
        """
        containers = restart_running_notebook.instance.spec.containers
        container_names = [container.name for container in containers]

        assert KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Pod '{restart_running_notebook.name}' missing '{KUBE_RBAC_PROXY_CONTAINER}' "
            f"after migration. Containers: {container_names}"
        )
        assert OAUTH_PROXY_CONTAINER not in container_names, (
            f"Pod '{restart_running_notebook.name}' still has '{OAUTH_PROXY_CONTAINER}' "
            f"after migration. Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_running_notebook_main_image_unchanged_after_restart(
        self,
        restart_running_notebook: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a running 2.x notebook is restarted after upgrade,
        When the controller migrates it to 3.x,
        Then the main container image should remain unchanged.
        """
        containers = restart_running_notebook.instance.spec.containers
        main_container = next(
            (
                container
                for container in containers
                if container.name not in (KUBE_RBAC_PROXY_CONTAINER, OAUTH_PROXY_CONTAINER)
            ),
            None,
        )
        assert main_container is not None, (
            f"No main container found in pod '{restart_running_notebook.name}'. "
            f"Containers: {[container.name for container in containers]}"
        )

        saved_image = upgrade_notebook_baseline.get("notebook_image")
        if saved_image:
            assert main_container.image == saved_image, (
                f"Main container image changed after migration. Expected: {saved_image}, got: {main_container.image}"
            )
