# Contributing guide

## Publishing a release

This project has a [GitHub Actions workflow](/.github/workflows/release.yaml) that publishes the `outpostkit` package to PyPI. The release process is triggered by manually creating and pushing a new git tag.

First, set the version number in [pyproject.toml](pyproject.toml) and commit it to the `main` branch:

```
version = "0.0.6"
```

Then run the following in your local checkout:

```sh
git checkout main
git fetch --all --tags
git tag 0.0.6
git push --tags
```

Then visit [github.com/outposthq/outpostkit-python/actions](https://github.com/outposthq/outpostkit-python/actions) to monitor the release process.
