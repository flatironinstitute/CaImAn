# CaImAn contributors guide
CaImAn is an open source project where *everyone* is welcome and encouraged to contribute. We have external contributors from all over the world, and we are always looking for more help. The guide below explains how to contribute to Caiman: if you have questions the process, please feel free to reach out at [Gitter](https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im). Everyone typically needs some help at first, so please don't be shy about reaching out. 

There are many different ways you can contribute to Caiman. The first and easiest way is to bring problems to our attention: if you find a bug, or think there is a feature that is lacking in Caiman, please bring it to our attention by [opening an issue at Github](https://github.com/flatironinstitute/CaImAn/issues). If you are a user, it also helps if you answer questions at issues or Gitter. 

Second, let's say you want to improve something yourself:

- Documentation like the one you are reading now
- The demo notebooks
- The code base

we welcome such contributions! To do this you need to change Caiman locally and then make a **Pull Request** (PR) at the repository. We will walk you through this process in the rest of the document. It may seem arcane at first, but once you have done it the first time, you will have the recipe and it will be easy to contribute.

Before you go through the work required to improve something, we recommend that you let us know your plans before putting in a lot of work. E.g., open an issue, or reach out on Gitter. This way, we can avoid duplicated effort if someone is already working on it, or  effort spent working on changes that may not be feasible. We can usually set up a video chat on Google or Skype to talk about a feature proposal if that works for you.

## Background: how does this work?
In this section we'll give general background on how making a contribution works in the context of working within a git repository. If you just want to get started quickly, feel free to skip to the next section.

There are many possible workflows for contributing to open source projects (see a summary [here](https://docs.gitlab.com/ee/topics/gitlab_flow.html)). The basic structure of the workflow we use at Caiman is illustrated here:

<img src="docs/img/gitflow.jpg">

When you are working on a git repository, you are always working on a *branch*, and all the branches have names. Some branches are special, like the *main* branch, which contains the stable version of the software for general release. Different feature branches, which will have names like `fix_plotting_bug` are usually short-lived and what individual contributors like you will work on. 

To keep a buffer between the main branch and individiaul feature branches, there is a second persistent branch, a development branch called `dev`. This the branch that contributors actively push their changes to, and is the one with all the newest developments. As some of the features are not fully tested, it is not encouraged for everyday use.

The workflow for contributing to Caiman is roughly illustrated by the numbers in the above diagram:
1) Create a feature branch from `dev`.
2) Work on that feature until you are ready to push it to Caiman.
3) Make a PR: this is when you request that your changes become merged into `dev` at Caiman. Note this merge won't be immediate, you will get feedback on your code, and probably be asked to make some changes. 
4) Eventually, once enough different new features accumulate in the `dev` branch to merit a new release (every month or so), the `dev` branch will be merged with `main`. This will become a new version of Caiman that people download when they run `mamba install caiman`. 

Below we have instructions on how to do all of the above steps, in addition to Step 0, which is how to set up a development environment. While all of this may seem like a lot, some of the steps are extremely simple. Also, it is a very rewarding experience to contribute to an open source project -- we hope you'll take the plunge!

## First, create a dedicated development environment
If you have downloaded Caiman for standard use, you probably installed it using `conda` as described on the README page. As a contributor, you will want to set up a dedicated development environment. This means you will be setting up a version of Caiman you will edit and tweak, uncoupled from your main installation that you use for everyday analysis. To set up a development environment so you can follow the worflow outlined above, do the following:

1. Fork and clone the caiman repository    
Go to the [https://github.com/flatironinstitute/CaImAn](Caiman repo) and hit the `Fork` button at the top right of the page. You now have Caiman on your own GitHub repo! On your computer, in your conda prompt, go to a directory where you want Caiman to download, and clone your personal Caiman repo: `git clone https://github.com/<your-username>/CaImAn.git` where <your-username> is replaced by your github username.
2. Install in dev mode   
In the previous step, you cloned your local fork of Caiman. Detailed instructions for installing in development mode can be found on our [installation page](./docs/source/Installation.rst). for both [Windows](https://github.com/flatironinstitute/CaImAn/blob/master/docs/source/Installation.rst#installation-on-windows) and [Mac/Linux](https://github.com/flatironinstitute/CaImAn/blob/master/docs/source/Installation.rst#installation-on-macos-and-linux). While there are some wrinkles, they both involve creating a conda environment from the `environment.yml` file you downloaded when you cloned the repo, activating that environment, and end with an 'in-place' install:

        pip install -e . 

    This command, performed within Caiman directory, installs Caiman directly from source code. Installing this way leaves all the files in that directory for you to modify as a contributor, which is quite different from what happens when you install the normal way. 

Note this section is partly based on the excellent [docs from Matplotlib](https://matplotlib.org/devdocs/devel/development_setup.html#installing-for-devs).


## Second, work on your feature branch
Once you have a development environment, most of the hard work is done. You can start working on changes. The main thing to remember is to follow the workflow in the diagram above. Let's say you want to work on feature `my_feature`. The first thing to do (label 1 in the diagram) is to create a new branch from the `dev` branch. Within the repo folder:

    git checkout -b my_feature dev

Then from within this branch you can do your usual work on the feature: work on stuff, add, commits, etc. This corresponds to the workflow in the blue boxes in the above diagram (label 2 in the figure).

If you changed anything in the `.py` modules, we request that you run tests on your code before making a PR. You can test the code by typing `caimanmanager.py test`. 

## Third, make a PR
Once your feature branch is ready,  it's time to make a PR (label 3 in the figure). This is fairly simple:

    git push -u origin my_feature

Then, go to your local fork of Caiman at your github repo, and a large green button will show prominently at the top, giving you the option of making a `Pull request`. When you click it, this will send over your feature for review (be sure to push to merge with dev branch, not main). 

PRs are reviewed by developers, and they will almost always give comments, making suggestions or asking questions about changes using Github's pull request interface. 

If conflicts emerge, either during tests or when our internal testing services run, we may ask you to resolve incompatiblities and update your PR. In such cases, when you work further on your branch and make more changes after making an initial PR, you will end up doing more commits and run the `push` command again. Conveniently, this will automatically push them to the same work-in-progress PR at Caiman. Note that if your PR is open for too long, merge conflicts can start to emerge as the `dev` branch changes.

## Fourth, wait for the work to show up in main
Once your work is done, it will be integrated into the main branch by the developers who maintain Caiman (label 4 in the figure). This is done every month or two, and is the stage when your work will actually be visible to people who download Caiman. It's at this point your name will appear when you click on the [list Contributors](https://github.com/flatironinstitute/CaImAn/graphs/contributors) at Github -- time to pat yourself on the back!

# What next?
Once you have gone through the above steps, you can delete your local feature branch. Before working on a new feature, you will want to make sure that your fork stays up to date with Caiman. You can do this easily with the user interface at GitHub (there is a button to sync up your repo with the original repository on a particular branch). There are a lot more details to git and GitHub workflows that we are leaving out: if you want to learn more, check out the following resources:

* [Getting started with git/github](https://github.com/EricThomson/git_learn)
* [GitHub on Contributing to a Project](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
* [GitHub skillbuilding](https://skills.github.com/)
* [Scipy git resources](https://docs.scipy.org/doc/scipy/dev/gitwash/gitwash.html#using-git)

Again, if you want to contribute and find any of the above bits confusing, please reach out at [Gitter](https://app.gitter.im/#/room/#agiovann_Constrained_NMF:gitter.im). We are here to help.





