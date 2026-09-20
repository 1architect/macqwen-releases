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

    def test_ci_and_release_automation_are_separate(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
        self.assertIn("models/flashnext", ci)
        self.assertIn("models/qwen27b", ci)
        self.assertIn("models/k2_horizon", ci)
        self.assertIn("models/bonsai2", ci)
        self.assertNotIn("gh release create", ci)
        self.assertIn("tags: ['v*']", release)
        self.assertIn("needs: test", release)
        self.assertIn("gh release create", release)


if __name__ == "__main__":
    unittest.main()
