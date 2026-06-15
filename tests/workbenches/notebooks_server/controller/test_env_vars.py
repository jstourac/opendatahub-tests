import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from pyhelper_utils.shell import run_command

from utilities.constants import Timeout

CUSTOM_ENV_VARS = [
    {"name": "MY_CUSTOM_VAR", "value": "hello-from-notebook"},
    {"name": "DATASET_PATH", "value": "/opt/app-root/data"},
    {"name": "DEBUG_MODE", "value": "true"},
]


class TestEnvironmentVariables:
    """Verify that environment variables from the Notebook CR propagate to the pod."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-envvars",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-envvars"},
                {
                    "namespace": "test-nb-envvars",
                    "name": "test-nb-envvars",
                    "extra_env_vars": CUSTOM_ENV_VARS,
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_custom_env_vars_in_pod_spec",
            )
        ],
        indirect=True,
    )
    def test_custom_env_vars_in_pod_spec(
        self,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify custom env vars from Notebook CR spec appear in the pod container spec.

        Given a Notebook CR with custom env vars in the container spec,
        When the controller creates the pod,
        Then the notebook container should have those env vars in its spec.
        """
        notebook_container = self._find_notebook_container(pod=notebook_pod, notebook_name=default_notebook.name)
        assert notebook_container, (
            f"Notebook container '{default_notebook.name}' not found in pod. "
            f"Available: {[c.name for c in notebook_pod.instance.spec.containers]}"
        )

        pod_env = {env.name: env.value for env in (notebook_container.env or [])}

        for expected_var in CUSTOM_ENV_VARS:
            var_name = expected_var["name"]
            var_value = expected_var["value"]
            assert var_name in pod_env, (
                f"Env var '{var_name}' not found in pod container spec. Present env vars: {list(pod_env.keys())}"
            )
            assert pod_env[var_name] == var_value, (
                f"Env var '{var_name}' value mismatch: expected '{var_value}', got '{pod_env[var_name]}'"
            )

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-envvars-exec",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-envvars-exec"},
                {
                    "namespace": "test-nb-envvars-exec",
                    "name": "test-nb-envvars-exec",
                    "extra_env_vars": CUSTOM_ENV_VARS,
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_custom_env_vars_visible_in_container",
            )
        ],
        indirect=True,
    )
    def test_custom_env_vars_visible_in_container(
        self,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify custom env vars are visible inside the running container via exec.

        Given a Notebook CR with custom env vars,
        When exec'ing printenv inside the notebook container,
        Then the custom env vars should be visible with correct values.
        """
        pod_name = notebook_pod.name
        namespace = notebook_pod.namespace
        container_name = default_notebook.name

        for expected_var in CUSTOM_ENV_VARS:
            var_name = expected_var["name"]
            var_value = expected_var["value"]

            cmd = [
                "oc",
                "exec",
                pod_name,
                "-n",
                namespace,
                "-c",
                container_name,
                "--",
                "printenv",
                var_name,
            ]
            exit_code, stdout, _ = run_command(
                command=cmd,
                verify_stderr=False,
                check=False,
            )

            assert exit_code == 0, f"Failed to exec printenv {var_name} in pod {pod_name}: exit code {exit_code}"
            assert stdout.strip() == var_value, (
                f"Env var '{var_name}' in container: expected '{var_value}', got '{stdout.strip()}'"
            )

    @staticmethod
    def _find_notebook_container(pod: Pod, notebook_name: str):
        """Find the notebook container in the pod by matching the notebook name."""
        for container in pod.instance.spec.containers:
            if container.name == notebook_name:
                return container
        return None
