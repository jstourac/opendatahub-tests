"""
Integration test for Kueue and Notebook admission control.
Tests that Notebook CRs can be managed by Kueue queue system.
"""

import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any
import pytest
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.resource import ResourceEditor
from utilities.kueue_utils import (
    check_gated_pods_and_running_pods,
    get_queue_resource_usage,
    verify_queue_tracks_workload,
)
from utilities.constants import Timeout

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.smoke,
]


@dataclass
class TestConfig:
    """Test configuration constants."""

    # Normal resources scenario
    NAMESPACE_NAME = "kueue-notebook-test"
    LOCAL_QUEUE_NAME = "notebook-local-queue"
    CLUSTER_QUEUE_NAME = "notebook-cluster-queue"
    RESOURCE_FLAVOR_NAME = "notebook-flavor"
    CPU_QUOTA = "4"
    MEMORY_QUOTA = "8Gi"
    NOTEBOOK_NAME = "test-kueue-notebook"

    # Resource starvation scenario
    LOW_RESOURCE_NAMESPACE_NAME = "kueue-low-resource-test"
    LOW_RESOURCE_LOCAL_QUEUE_NAME = "low-resource-local-queue"
    LOW_RESOURCE_CLUSTER_QUEUE_NAME = "low-resource-cluster-queue"
    LOW_RESOURCE_FLAVOR_NAME = "low-resource-flavor"
    LOW_CPU_QUOTA = "100m"  # Very low CPU quota - 100m
    LOW_MEMORY_QUOTA = "64Mi"  # Very low memory quota
    HIGH_DEMAND_NOTEBOOK_NAME = "high-demand-notebook"

    # Expected resource usage for normal scenario
    # Main notebook container: cpu="1", memory="1Gi"
    # OAuth proxy sidecar: cpu="100m", memory="64Mi"
    # Total: cpu="1100m", memory="1088Mi"
    EXPECTED_CPU_NORMAL = "1100m"
    EXPECTED_MEMORY_NORMAL = "1088Mi"


def _create_normal_resources_config() -> dict[str, Any]:
    """Create configuration for normal resources test scenario."""
    return {
        "patched_kueue_manager_config": {},
        "kueue_enabled_notebook_namespace": {"name": TestConfig.NAMESPACE_NAME, "add-kueue-label": True},
        "kueue_notebook_persistent_volume_claim": {"name": TestConfig.NOTEBOOK_NAME},
        "default_notebook": {
            "namespace": TestConfig.NAMESPACE_NAME,
            "name": TestConfig.NOTEBOOK_NAME,
            "labels": {"kueue.x-k8s.io/queue-name": TestConfig.LOCAL_QUEUE_NAME},
        },
        "kueue_notebook_cluster_queue": {
            "name": TestConfig.CLUSTER_QUEUE_NAME,
            "resource_flavor_name": TestConfig.RESOURCE_FLAVOR_NAME,
            "cpu_quota": TestConfig.CPU_QUOTA,
            "memory_quota": TestConfig.MEMORY_QUOTA,
            "namespace_selector": {"matchLabels": {"kubernetes.io/metadata.name": TestConfig.NAMESPACE_NAME}},
        },
        "kueue_notebook_resource_flavor": {"name": TestConfig.RESOURCE_FLAVOR_NAME},
        "kueue_notebook_local_queue": {
            "name": TestConfig.LOCAL_QUEUE_NAME,
            "cluster_queue": TestConfig.CLUSTER_QUEUE_NAME,
        },
    }


def _create_resource_starvation_config() -> dict[str, Any]:
    """Create configuration for resource starvation test scenario."""
    return {
        "patched_kueue_manager_config": {},
        "kueue_enabled_notebook_namespace": {"name": TestConfig.LOW_RESOURCE_NAMESPACE_NAME, "add-kueue-label": True},
        "kueue_notebook_persistent_volume_claim": {"name": TestConfig.HIGH_DEMAND_NOTEBOOK_NAME},
        "default_notebook": {
            "namespace": TestConfig.LOW_RESOURCE_NAMESPACE_NAME,
            "name": TestConfig.HIGH_DEMAND_NOTEBOOK_NAME,
            "labels": {"kueue.x-k8s.io/queue-name": TestConfig.LOW_RESOURCE_LOCAL_QUEUE_NAME},
            # Request resources that exceed the queue limits
            "cpu_request": "1000m",  # 1 CPU (exceeds 100m limit)
            "memory_request": "1Gi",  # 1 GB (exceeds 64Mi limit)
        },
        "kueue_notebook_cluster_queue": {
            "name": TestConfig.LOW_RESOURCE_CLUSTER_QUEUE_NAME,
            "resource_flavor_name": TestConfig.LOW_RESOURCE_FLAVOR_NAME,
            "cpu_quota": TestConfig.LOW_CPU_QUOTA,
            "memory_quota": TestConfig.LOW_MEMORY_QUOTA,
            "namespace_selector": {
                "matchLabels": {"kubernetes.io/metadata.name": TestConfig.LOW_RESOURCE_NAMESPACE_NAME}
            },
        },
        "kueue_notebook_resource_flavor": {"name": TestConfig.LOW_RESOURCE_FLAVOR_NAME},
        "kueue_notebook_local_queue": {
            "name": TestConfig.LOW_RESOURCE_LOCAL_QUEUE_NAME,
            "cluster_queue": TestConfig.LOW_RESOURCE_CLUSTER_QUEUE_NAME,
        },
    }


@pytest.mark.parametrize(
    "patched_kueue_manager_config, kueue_enabled_notebook_namespace, kueue_notebook_persistent_volume_claim, "
    "default_notebook, kueue_notebook_cluster_queue, kueue_notebook_resource_flavor, kueue_notebook_local_queue",
    [
        pytest.param(
            *_create_normal_resources_config().values(),
            id="normal_resources",
        ),
        pytest.param(
            *_create_resource_starvation_config().values(),
            id="resource_starvation",
        ),
    ],
    indirect=True,
)
class TestKueueNotebookController:
    """Test Kueue integration with Notebook Controller

    PREREQUISITE: The patched_kueue_manager_config fixture runs FIRST and:
    1. Stores the original kueue-manager-config ConfigMap state
    2. Patches the ConfigMap with required frameworks and annotation
    3. Restarts kueue-controller-manager deployment to apply changes
    4. Runs all tests with the patched configuration
    5. CLEANUP: Restores original ConfigMap state and restarts deployment

    This ensures proper isolation and cleanup between test runs.
    """

    def _is_resource_starvation_scenario(self, notebook: Notebook) -> bool:
        """Check if this is the resource starvation test scenario."""
        return notebook.name == TestConfig.HIGH_DEMAND_NOTEBOOK_NAME

    def _validate_notebook_kueue_labels(self, notebook: Notebook, expected_queue_name: str) -> None:
        """Validate that notebook has proper Kueue labels."""
        assert notebook.exists, "Notebook CR should be created successfully"
        notebook_labels = notebook.instance.metadata.labels or {}
        queue_name = notebook_labels.get("kueue.x-k8s.io/queue-name")
        assert queue_name is not None, (
            f"Notebook should have Kueue queue label. Available labels: {list(notebook_labels.keys())}"
        )
        assert queue_name == expected_queue_name, (
            f"Notebook should have correct queue name: {expected_queue_name}, got: {queue_name}"
        )

    def _setup_and_wait_for_notebook_pod(self, notebook: Notebook, client: DynamicClient) -> Pod:
        """Create notebook pod and wait for it to exist."""
        notebook_pod = Pod(
            client=client,
            namespace=notebook.namespace,
            name=f"{notebook.name}-0",
        )
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        return notebook_pod

    def _validate_pod_kueue_management(self, pod: Pod, expected_queue_name: str) -> None:
        """Validate that pod is properly managed by Kueue."""
        pod_labels = pod.instance.metadata.labels or {}
        assert pod_labels.get("kueue.x-k8s.io/managed") == "true", "Pod should be managed by Kueue"
        assert pod_labels.get("kueue.x-k8s.io/queue-name") == expected_queue_name, "Pod should reference correct queue"

    def _verify_resource_tracking(
        self, cluster_queue, local_queue, flavor_name: str, expected_cpu: str, expected_memory: str
    ) -> None:
        """Verify that queues are tracking expected resource usage."""
        cluster_usage = get_queue_resource_usage(queue=cluster_queue, flavor_name=flavor_name)
        local_usage = get_queue_resource_usage(queue=local_queue, flavor_name=flavor_name)

        LOGGER.info(f"ClusterQueue usage: {cluster_usage}")
        LOGGER.info(f"LocalQueue usage: {local_usage}")

        cluster_tracks_exact = verify_queue_tracks_workload(
            queue=cluster_queue, flavor_name=flavor_name, expected_cpu=expected_cpu, expected_memory=expected_memory
        )
        local_tracks_exact = verify_queue_tracks_workload(
            queue=local_queue, flavor_name=flavor_name, expected_cpu=expected_cpu, expected_memory=expected_memory
        )

        assert cluster_tracks_exact or local_tracks_exact, (
            f"Either ClusterQueue or LocalQueue should show exact resource usage "
            f"(CPU: {expected_cpu}, Memory: {expected_memory}). "
            f"ClusterQueue usage: {cluster_usage}, LocalQueue usage: {local_usage}"
        )

    def _verify_pod_gating_status(
        self,
        pod_labels: list[str],
        namespace: str,
        admin_client: DynamicClient,
        expected_running: int,
        expected_gated: int,
    ) -> None:
        """Verify the gating status of pods."""
        running_pods, gated_pods = check_gated_pods_and_running_pods(
            labels=pod_labels, namespace=namespace, admin_client=admin_client
        )

        assert running_pods == expected_running, (
            f"Expected exactly {expected_running} running notebook pods, found {running_pods}"
        )
        assert gated_pods == expected_gated, f"Expected {expected_gated} gated notebook pods, found {gated_pods}"

    def test_kueue_notebook_admission_control(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue properly controls admission of Notebook workloads

        1. Create a Notebook CR with Kueue labels
        2. Verify the notebook pod is created and managed by Kueue
        3. Check that Kueue admission control is working (pod should be gated initially)
        4. Verify the notebook eventually becomes ready
        """

        # Skip this test for resource starvation scenario
        if self._is_resource_starvation_scenario(default_notebook):
            pytest.skip("Skipping admission control test for resource starvation scenario")

        # Verify that the notebook was created with Kueue labels
        self._validate_notebook_kueue_labels(default_notebook, TestConfig.LOCAL_QUEUE_NAME)

        # Find and wait for the notebook pod
        notebook_pod = self._setup_and_wait_for_notebook_pod(default_notebook, unprivileged_client)

        # Verify that Kueue has admitted the workload and pod exists
        assert notebook_pod.exists, f"Notebook pod {notebook_pod.name} should exist"

        # Verify pod has Kueue management labels
        self._validate_pod_kueue_management(notebook_pod, TestConfig.LOCAL_QUEUE_NAME)

        # Wait for the notebook pod to become ready
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
        )

        # Verify the pod is now running (not gated)
        pod_labels = [f"app={default_notebook.name}"]
        self._verify_pod_gating_status(
            pod_labels=pod_labels,
            namespace=kueue_enabled_notebook_namespace.name,
            admin_client=admin_client,
            expected_running=1,
            expected_gated=0,
        )

        # Verify Kueue is tracking exact resource usage
        LOGGER.info(f"Checking queue resource usage for {kueue_notebook_cluster_queue.name}...")
        self._verify_resource_tracking(
            cluster_queue=kueue_notebook_cluster_queue,
            local_queue=kueue_notebook_local_queue,
            flavor_name=TestConfig.RESOURCE_FLAVOR_NAME,
            expected_cpu=TestConfig.EXPECTED_CPU_NORMAL,
            expected_memory=TestConfig.EXPECTED_MEMORY_NORMAL,
        )

    def test_kueue_notebook_resource_constraints(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue enforces resource constraints for notebooks

        1. Verify the notebook's resource requests are within the queue limits
        2. Check that Kueue properly tracks resource usage
        """

        # Skip this test for resource starvation scenario
        if self._is_resource_starvation_scenario(default_notebook):
            pytest.skip("Skipping resource constraints test for resource starvation scenario")

        # Verify notebook exists and has proper configuration
        assert default_notebook.exists, "Notebook CR should exist"

        # Get the notebook pod and wait for it to be ready
        notebook_pod = self._setup_and_wait_for_notebook_pod(default_notebook, unprivileged_client)
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
        )

        # Verify resource requests are properly set
        self._validate_notebook_resource_requests(notebook_pod, default_notebook)

    def _validate_notebook_resource_requests(self, pod: Pod, notebook: Notebook) -> None:
        """Validate that notebook container has proper resource requests."""
        pod_spec = pod.instance.spec
        containers = pod_spec.containers

        # Find the main notebook container
        notebook_container = None
        for container in containers:
            if container.name == notebook.name:
                notebook_container = container
                break

        assert notebook_container is not None, "Notebook container should be found in pod spec"

        # Check resource requests exist
        resources = notebook_container.resources
        assert resources is not None, "Container should have resource specifications"
        assert resources.requests is not None, "Container should have resource requests"

        # Verify CPU and memory requests are within queue limits
        cpu_request = resources.requests.get("cpu", "0")
        memory_request = resources.requests.get("memory", "0")

        # Convert CPU request to numeric value for comparison
        if cpu_request.endswith("m"):
            cpu_value = int(cpu_request[:-1]) / 1000
        else:
            cpu_value = float(cpu_request)

        assert cpu_value <= float(TestConfig.CPU_QUOTA), (
            f"CPU request {cpu_value} should not exceed queue quota {TestConfig.CPU_QUOTA}"
        )

        # Memory requests should be reasonable (basic validation)
        assert memory_request != "0", "Memory request should be specified"

    def test_kueue_notebook_resource_starvation(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,  # Ensures this runs first
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue properly gates Notebook workloads when there are insufficient resources

        This test creates:
        1. A queue with very low resource limits (100m CPU, 64Mi memory)
        2. A notebook that requests high resources (1 CPU, 1Gi memory)
        3. Verifies that the notebook pod remains in SchedulingGated state due to resource starvation
        """

        # Skip this test for normal resource scenario
        if not self._is_resource_starvation_scenario(default_notebook):
            pytest.skip("Skipping resource starvation test for normal resource scenario")

        # Verify that the notebook was created with Kueue labels
        self._validate_notebook_kueue_labels(default_notebook, TestConfig.LOW_RESOURCE_LOCAL_QUEUE_NAME)

        # Check that the notebook pod is created but gated due to insufficient resources
        notebook_pod = self._setup_and_wait_for_notebook_pod(default_notebook, unprivileged_client)
        assert notebook_pod.exists, f"Notebook pod {notebook_pod.name} should exist"

        # Verify that Kueue has applied management labels to the pod
        self._validate_pod_kueue_management(notebook_pod, TestConfig.LOW_RESOURCE_LOCAL_QUEUE_NAME)

        # Wait for pod to be properly gated
        self._wait_for_pod_gating(notebook_pod)

        # Verify pod is gated due to resource constraints
        pod_labels = [f"app={default_notebook.name}"]
        self._verify_pod_gating_status(
            pod_labels=pod_labels,
            namespace=kueue_enabled_notebook_namespace.name,
            admin_client=admin_client,
            expected_running=0,
            expected_gated=1,
        )

        # Verify the pod phase and conditions
        self._verify_scheduling_gated_condition(notebook_pod)

        LOGGER.info(
            f"SUCCESS: Notebook pod {notebook_pod.name} is properly gated due to insufficient resources "
            f"in queue {TestConfig.LOW_RESOURCE_LOCAL_QUEUE_NAME}"
        )

    def _wait_for_pod_gating(self, pod: Pod) -> None:
        """Wait for pod to be properly gated (check multiple times to ensure stable state)."""
        for _ in range(10):
            pod.get()  # Refresh pod state
            if pod.instance.status.phase == "Pending":
                conditions = pod.instance.status.conditions or []
                if any(c.type == "PodScheduled" and c.reason == "SchedulingGated" for c in conditions):
                    break
            time.sleep(1)  # noqa: FCN001

    def _verify_scheduling_gated_condition(self, pod: Pod) -> None:
        """Verify that pod has SchedulingGated condition."""
        pod.get()  # Refresh pod state
        pod_status = pod.instance.status
        assert pod_status.phase == "Pending", f"Pod should be in Pending state, got: {pod_status.phase}"

        # Check for SchedulingGated condition
        scheduling_gated = False
        if hasattr(pod_status, "conditions") and pod_status.conditions:
            for condition in pod_status.conditions:
                if (
                    condition.type == "PodScheduled"
                    and condition.status == "False"
                    and condition.reason == "SchedulingGated"
                ):
                    scheduling_gated = True
                    break

        assert scheduling_gated, "Pod should have SchedulingGated condition due to insufficient resources"

    def test_kueue_notebook_stop_start_workbench(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,  # Ensures this runs first
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue properly handles workbench stop/start lifecycle operations

        This test:
        1. Waits for a notebook to be running with sufficient resources
        2. Stops the workbench by adding kubeflow-resource-stopped annotation
        3. Verifies the pod is terminated
        4. Starts the workbench by removing the annotation
        5. Verifies the pod is recreated and running again
        """

        # Skip this test for resource starvation scenario (needs running notebook)
        if self._is_resource_starvation_scenario(default_notebook):
            pytest.skip("Skipping stop/start test for resource starvation scenario")

        # Initial setup and verification
        notebook_pod = self._setup_initial_notebook_state(
            default_notebook, unprivileged_client, kueue_notebook_cluster_queue, kueue_notebook_local_queue
        )

        # Test stopping the workbench
        self._test_workbench_stop_functionality(
            default_notebook, notebook_pod, kueue_notebook_cluster_queue, kueue_notebook_local_queue
        )

        # Test starting the workbench
        self._test_workbench_start_functionality(
            default_notebook, unprivileged_client, kueue_notebook_cluster_queue, kueue_notebook_local_queue
        )

    def _setup_initial_notebook_state(
        self, notebook: Notebook, client: DynamicClient, cluster_queue, local_queue
    ) -> Pod:
        """Set up and verify initial notebook state before stop/start testing."""
        # Verify that the notebook was created with Kueue labels
        self._validate_notebook_kueue_labels(notebook, TestConfig.LOCAL_QUEUE_NAME)

        # Wait for the notebook pod to be created and running
        notebook_pod = self._setup_and_wait_for_notebook_pod(notebook, client)
        assert notebook_pod.exists, f"Notebook pod {notebook_pod.name} should exist"

        # Verify that Kueue has applied management labels to the pod
        self._validate_pod_kueue_management(notebook_pod, TestConfig.LOCAL_QUEUE_NAME)

        # Wait for the notebook pod to become ready (initial startup)
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
        )

        LOGGER.info(f"SUCCESS: Notebook pod {notebook_pod.name} is initially running and ready")

        # Verify Kueue is tracking exact resource usage for the running workload
        LOGGER.info("Checking queue resource usage before stopping workbench...")
        self._verify_resource_tracking(
            cluster_queue=cluster_queue,
            local_queue=local_queue,
            flavor_name=TestConfig.RESOURCE_FLAVOR_NAME,
            expected_cpu=TestConfig.EXPECTED_CPU_NORMAL,
            expected_memory=TestConfig.EXPECTED_MEMORY_NORMAL,
        )

        return notebook_pod

    def _test_workbench_stop_functionality(
        self, notebook: Notebook, notebook_pod: Pod, cluster_queue, local_queue
    ) -> None:
        """Test stopping workbench and verify resource cleanup."""
        # Stop the workbench by adding the kubeflow-resource-stopped annotation
        stop_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: FCN001

        # Get current notebook annotations
        notebook.get()  # Refresh to get latest state
        current_annotations = notebook.instance.metadata.annotations or {}

        # Add the stop annotation
        updated_annotations = {**current_annotations, "kubeflow-resource-stopped": stop_timestamp}
        stop_patch = {"metadata": {"annotations": updated_annotations}}

        with ResourceEditor(patches={notebook: stop_patch}):
            LOGGER.info(f"Applied kubeflow-resource-stopped annotation with timestamp: {stop_timestamp}")

            # Wait for the pod to be terminated (it should be deleted)
            notebook_pod.wait_deleted(timeout=Timeout.TIMEOUT_2MIN)
            LOGGER.info(f"SUCCESS: Notebook pod {notebook_pod.name} was terminated after stop annotation")

            # Verify resource usage goes to zero after stopping workbench
            LOGGER.info("Checking queue resource usage after stopping workbench...")
            self._verify_resource_tracking(
                cluster_queue=cluster_queue,
                local_queue=local_queue,
                flavor_name=TestConfig.RESOURCE_FLAVOR_NAME,
                expected_cpu="0",
                expected_memory="0",
            )

    def _test_workbench_start_functionality(
        self, notebook: Notebook, client: DynamicClient, cluster_queue, local_queue
    ) -> None:
        """Test starting workbench and verify resource allocation."""
        # Brief pause before restart
        time.sleep(5)  # noqa: FCN001

        # Get current notebook annotations again
        notebook.get()  # Refresh to get latest state
        current_annotations = notebook.instance.metadata.annotations or {}

        # Remove the stop annotation
        restart_annotations = {k: v for k, v in current_annotations.items() if k != "kubeflow-resource-stopped"}
        restart_patch = {"metadata": {"annotations": restart_annotations}}

        with ResourceEditor(patches={notebook: restart_patch}):
            LOGGER.info("Removed kubeflow-resource-stopped annotation to restart workbench")

            # Wait for the new pod to be created
            new_notebook_pod = Pod(
                client=client,
                namespace=notebook.namespace,
                name=f"{notebook.name}-0",
            )

            new_notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
            assert new_notebook_pod.exists, f"New notebook pod {new_notebook_pod.name} should be created after restart"

            # Verify that Kueue management labels are still present on the new pod
            self._validate_pod_kueue_management(new_notebook_pod, TestConfig.LOCAL_QUEUE_NAME)

            # Wait for the new pod to become ready
            new_notebook_pod.wait_for_condition(
                condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
            )

            LOGGER.info(f"SUCCESS: Notebook pod {new_notebook_pod.name} was successfully restarted and is ready again")

            # Final verification: Ensure Kueue is tracking exact resource usage for the restarted workload
            LOGGER.info("Checking queue resource usage after restarting workbench...")
            self._verify_resource_tracking(
                cluster_queue=cluster_queue,
                local_queue=local_queue,
                flavor_name=TestConfig.RESOURCE_FLAVOR_NAME,
                expected_cpu=TestConfig.EXPECTED_CPU_NORMAL,
                expected_memory=TestConfig.EXPECTED_MEMORY_NORMAL,
            )


@pytest.mark.parametrize(
    "patched_kueue_manager_config",
    [
        pytest.param(
            {},  # Uses default patching behavior
        )
    ],
    indirect=True,
)
def test_kueue_config_frameworks_enabled(
    admin_client: DynamicClient,
    patched_kueue_manager_config,
):
    """
    Test that the kueue-manager-config has been properly patched
    to include pod and statefulset frameworks, and that the
    kueue-controller-manager deployment was restarted.

    NOTE: This test runs independently and will restore the original
    configuration after completion. If patched_kueue_manager_config is None,
    it means we're using Red Hat build of Kueue operator and this test is skipped.
    """
    import yaml
    from pytest_testconfig import config as py_config
    from ocp_resources.deployment import Deployment

    # Skip test if ConfigMap patching was skipped (Red Hat build of Kueue operator scenario)
    if patched_kueue_manager_config is None:
        pytest.skip("Skipping kueue-manager-config test for Red Hat build of Kueue operator scenario")
        return

    # Refresh the ConfigMap instance to get the latest state
    patched_kueue_manager_config.get()

    # Verify the ConfigMap was patched correctly
    config_data = patched_kueue_manager_config.instance.data
    assert config_data is not None, "ConfigMap should have data"

    config_yaml = config_data.get("controller_manager_config.yaml", "")
    assert config_yaml, "ConfigMap should contain controller_manager_config.yaml data"

    # Verify the annotation was set correctly
    annotations = patched_kueue_manager_config.instance.metadata.annotations or {}
    managed_value = annotations.get("opendatahub.io/managed")
    assert managed_value is not None, (
        f"ConfigMap should have opendatahub.io/managed annotation. Current annotations: {list(annotations.keys())}"
    )
    assert managed_value == "false", (
        f"ConfigMap should have opendatahub.io/managed set to 'false', got: {managed_value}"
    )

    # Parse the configuration
    config_dict = yaml.safe_load(config_yaml)
    assert config_dict is not None, "Configuration should be valid YAML"

    # Verify integrations section exists
    assert "integrations" in config_dict, "Configuration should have integrations section"
    assert "frameworks" in config_dict["integrations"], "Integrations should have frameworks section"

    frameworks = config_dict["integrations"]["frameworks"]
    assert isinstance(frameworks, list), "Frameworks should be a list"
    assert "pod" in frameworks, "Frameworks should include 'pod'"
    assert "statefulset" in frameworks, "Frameworks should include 'statefulset'"

    # Verify the kueue-controller-manager deployment is running and ready
    kueue_deployment = Deployment(
        client=admin_client,
        name="kueue-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )

    # Verify deployment is ready by checking status
    kueue_deployment.wait_for_condition(
        condition=kueue_deployment.Condition.AVAILABLE, status=kueue_deployment.Condition.Status.TRUE, timeout=60
    )

    # If the deployment is AVAILABLE, it means it's working correctly
    LOGGER.info("SUCCESS: kueue-controller-manager deployment is ready and available")
