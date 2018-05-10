This is a guide to contributing to CaImAn. We have external contributors all over the world.

Switch to the dev branch
========================
Active development on CaImAn happens on the dev branch, and occasionally we sync changes over to the master branch. We would prefer your pull requests be against the dev branch, and be directed against it.

For rapid development, do an in-place install
=============================================
People who are not editing CaImAn sources should do an ordinary install with "pip install ." ; people who *are* actively changing CaImAn may find this inconvenient because they don't want to continually uninstall and reinstall the package. They should instead install with "pip install -e ." which performs a "symlink install". If they are editing datafiles that normally go in the ~/caiman_data directory, they might also consider setting the CAIMAN_DATA environment variable to their source dir, so they don't need to use the caimandata.py command to continually uninstall/reinstall the datadir.

Pull Requests
=============
We use github to host the CaImAn source tree, and contributions should come in the form of pull requests. Please do not email us diffs or replacement files. We also may do code review, making suggestions or asking questions about a diff using Github's pull request interface. Please be willing to chat with us about your proposed changes.

Your code should merge cleanly into the codebase; if it does not, you should rebase it, resolve any incompatibilities, and update your PR. Committing to the branch where your PR is committed will do this for you automatically. Note that if your PR is open for too long, merge conflicts will happen as upstream dev changes.

Test your code first
====================
The caimandata command has code to run the internal test suite on your platform. Please do this for whatever platform(s) you have access to; our Jenkins is not public and it's easier for you to find problems this way than to ask for text logs to be sent to you.

Platforms we care about
=======================
CaImAn code runs:
* On Linux (well supported)
* On MacOS (well supported)
* On Windows (reasonably well supported)

And on:
* Python 3 (primary target)
* Python 2 (secondary target)

Your code should be portable to the platforms above, and we have test suites that provide coverage for this.

Talk to us
==========
It's helpful, if you're starting to put together a diff, to let us know what you're working on. With this, we can avoid duplicated effort, or wasted effort from changes that we might not want to accept. We can usually setup a videoconference (Google Hangouts, Skype) if that works for you. We also like hearing from you on Gitter (link is in the main readme).
