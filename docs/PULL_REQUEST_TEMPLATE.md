# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

# Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would break back-compatibility)
- [ ] Changes in documentation or new examples for demos and/or use cases.

# Branching
- All PRs should be made against the **dev** branch. The main branch is not often merged back to dev.
- If you want to get your PR out to the world faster (urgent bugfix), poke pgunn to cut a release; this will get it onto github and into conda faster

# Has your PR been tested?

If you're fixing a bug or introducing a new feature it is recommended you run the tests by typing

```caimanmanager test```

and

```caimanmanager demotest```

prior to submitting your pull request. 

Please describe any additional tests that you ran to verify your changes. If they are fast you can also
include them in the folder 'caiman/tests/` and name them `test_***.py` so they can be included in our lists of tests.
