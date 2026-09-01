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

    def test_pinned_dependencies_match_the_requirements_file(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        packaged = set(data["project"]["optional-dependencies"]["flashnext"])
        requirements = {
            line.strip() for line in (ROOT / "requirements-flashnext.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertEqual(packaged, requirements)

    def test_ci_and_release_automation_exist(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        self.assertIn("models/flashnext", workflow)
        self.assertIn("gh release create", workflow)


if __name__ == "__main__":
    unittest.main()
