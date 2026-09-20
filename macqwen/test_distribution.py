from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_REPOSITORY = "https://github.com/1architect/macqwen-releases"


class DistributionTests(unittest.TestCase):
    def test_readme_uses_the_public_clone_url(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn(CANONICAL_REPOSITORY + ".git", readme)
        self.assertNotIn("github.com/1architect/MACQWEN", readme)

    def test_package_metadata_uses_the_public_repository(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(data["project"]["urls"]["Repository"], CANONICAL_REPOSITORY)
        changelog = (ROOT / "CHANGELOG.md").read_text()
        self.assertIn(f"## MACQWEN {data['project']['version']} ", changelog)

    def test_shared_runtime_dependencies_are_pinned_in_project_metadata(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            set(data["project"]["dependencies"]),
            {
                "mlx==0.32.2",
                "mlx-lm==0.31.3",
                "mlx-vlm==0.6.17",
                "transformers==5.16.1",
                "numpy==2.5.2",
                "requests==2.34.2",
                "huggingface-hub==1.29.0",
            },
        )

    def test_ci_and_release_automation_exist(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        self.assertIn("models/flashnext", workflow)
        self.assertIn("models/qwen27b", workflow)
        self.assertIn("models/k2_horizon", workflow)
        self.assertIn("models/bonsai2", workflow)
        self.assertIn("gh release create", workflow)


if __name__ == "__main__":
    unittest.main()
