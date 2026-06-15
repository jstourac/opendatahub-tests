import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod

from utilities.constants import Timeout


class TestNotebookResourceLimits:
    """Verify that resource requests/limits on the Notebook CR propagate to the pod."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-resources-small",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-resources-small"},
                {
                    "namespace": "test-nb-resources-small",
                    "name": "test-nb-resources-small",
                    "resources": {
                        "limits": {"cpu": "1", "memory": "2Gi"},
                        "requests": {"cpu": "500m", "memory": "512Mi"},
                    },
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_small_profile_resources",
            )
        ],
        indirect=True,
    )
    def test_notebook_pod_resources_match_spec(
        self,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify notebook pod container resources match the Notebook CR spec.

        Given a Notebook CR is created with explicit resource requests and limits,
        When the controller reconciles and the pod becomes Ready,
        Then the notebook container's resources should exactly match the CR spec.
        """
        notebook_container = self._find_notebook_container(pod=notebook_pod, notebook_name=default_notebook.name)
        assert notebook_container, (
            f"Notebook container '{default_notebook.name}' not found in pod. "
            f"Available: {[c.name for c in notebook_pod.instance.spec.containers]}"
        )

        expected_limits = {"cpu": "1", "memory": "2Gi"}
        expected_requests = {"cpu": "500m", "memory": "512Mi"}

        actual_limits = dict(notebook_container.resources.limits or {})
        actual_requests = dict(notebook_container.resources.requests or {})

        assert actual_limits["cpu"] == expected_limits["cpu"], (
            f"CPU limit mismatch: expected '{expected_limits['cpu']}', got '{actual_limits.get('cpu')}'"
        )
        assert actual_limits["memory"] == expected_limits["memory"], (
            f"Memory limit mismatch: expected '{expected_limits['memory']}', got '{actual_limits.get('memory')}'"
        )
        assert actual_requests["cpu"] == expected_requests["cpu"], (
            f"CPU request mismatch: expected '{expected_requests['cpu']}', got '{actual_requests.get('cpu')}'"
        )
        assert actual_requests["memory"] == expected_requests["memory"], (
            f"Memory request mismatch: expected '{expected_requests['memory']}', got '{actual_requests.get('memory')}'"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-resources-large",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-resources-large"},
                {
                    "namespace": "test-nb-resources-large",
                    "name": "test-nb-resources-large",
                    "resources": {
                        "limits": {"cpu": "4", "memory": "8Gi"},
                        "requests": {"cpu": "2", "memory": "4Gi"},
                    },
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_large_profile_resources",
            )
        ],
        indirect=True,
    )
    def test_notebook_pod_large_profile_resources(
        self,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify a larger hardware profile propagates correctly to the pod.

        Given a Notebook CR with a large resource profile (4 CPU, 8Gi memory),
        When the controller reconciles and the pod becomes Ready,
        Then the notebook container's resources should reflect the large profile.
        """
        notebook_container = self._find_notebook_container(pod=notebook_pod, notebook_name=default_notebook.name)
        assert notebook_container, (
            f"Notebook container '{default_notebook.name}' not found in pod. "
            f"Available: {[c.name for c in notebook_pod.instance.spec.containers]}"
        )

        expected_limits = {"cpu": "4", "memory": "8Gi"}
        expected_requests = {"cpu": "2", "memory": "4Gi"}

        actual_limits = dict(notebook_container.resources.limits or {})
        actual_requests = dict(notebook_container.resources.requests or {})

        assert actual_limits["cpu"] == expected_limits["cpu"], (
            f"CPU limit mismatch: expected '{expected_limits['cpu']}', got '{actual_limits.get('cpu')}'"
        )
        assert actual_limits["memory"] == expected_limits["memory"], (
            f"Memory limit mismatch: expected '{expected_limits['memory']}', got '{actual_limits.get('memory')}'"
        )
        assert actual_requests["cpu"] == expected_requests["cpu"], (
            f"CPU request mismatch: expected '{expected_requests['cpu']}', got '{actual_requests.get('cpu')}'"
        )
        assert actual_requests["memory"] == expected_requests["memory"], (
            f"Memory request mismatch: expected '{expected_requests['memory']}', got '{actual_requests.get('memory')}'"
        )

    @staticmethod
    def _find_notebook_container(pod: Pod, notebook_name: str):
        """Find the notebook container in the pod by matching the notebook name."""
        for container in pod.instance.spec.containers:
            if container.name == notebook_name:
                return container
        return None
