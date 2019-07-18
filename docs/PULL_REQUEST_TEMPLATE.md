# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Changes in documentation or new examples for demos and/or use cases.

## Branching

- Pull requests with non breaking bug fixes should be made against the **master** branch.
- All other PRs should be made against the **dev** branch.

# How Has This Been Tested?

If you're fixing a bug or introducing a new feauter make sure you run the tests by typing

```python caimanmanager.py test```

and

```python caimanmanager.py demotest```

prior to submitting your pull request. 

Please describe any additional tests that you ran to verify your changes. If they are fast you can also
include them in the folder 'caiman/tests/` and name them `test_***.py` so they can be included in our lists of tests.

# Checklist (check all that apply):

- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes in the docstrings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules
- [ ] There are no conflicts with the existing branch
