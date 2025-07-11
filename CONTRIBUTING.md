# Contributing to GNU Radio
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing. These are mostly guidelines, not rules. Use your best judgement, and feel free to propose changes to this document in a pull request.

## Open an issue
For bugs, issues, or other discussion, please log a new issue in the GitHub repository.

GitHub supports [markdown](https://help.github.com/categories/writing-on-github/), so when filing bugs make sure you check the formatting before clicking submit.

## Other discussions
For general "how-to" and guidance questions about using GNU Radio to build and run applications, please have a look at the various
examples or if you cannot find anything that fits your use-case use GitHub's discussions forum.

## Contributing code and content
We welcome all forms of contributions from the community. Please read the following guidelines to maximise the chances of your PR being merged.

### Communication
 - Before starting work on a feature, check if there isn't already an examples in the 'samples' sub-module.
   If not, then please open an issue on GitHub describing the proposed feature. We want to make sure any feature work goes smoothly. 
   We're happy to work with you to determine if it fits the current project direction and make sure no one else is already working on it.

 - For any work related to setting up build, test, and CI for GNU Radio on GitHub, or for small patches or bug fixes, please open an issue
   for tracking purposes, but we generally don't need a discussion prior to opening a PR.

### Development process
Please be sure to follow the usual process for submitting PRs:

 - Fork the repository
 - Make sure it compiles w/o errors against the current release 'main' branch:
 - Write and add a descriptive/meaningful unit-test-case
 - apply the default code formatter (to minimise future refactoring) 
 - Please check against common sanitizers, the CI/CD pipeline, or similar other QA code checker (N.B. other/further code improvements are welcome)
 - Create a pull request 
 - Make sure your PR title is descriptive
 - Include a link back to an open issue in the PR description

We reserve the right to close PRs that are not making progress. Closed PRs can be reopened again later and work can resume.

### Copyright Assignment

GNU Radio does not claim ownership of any contributions you make. All copyrights remain with the original author of the contribution. As such, we don't require copyright notices in the header of each file, and the broader copyright statement for collective attribution is located in the [README](README.md)


#### Non-Revocability of Contributions

By submitting a contribution to the GNU Radio project, you agree that your contribution is non-revocable. Once a contribution is accepted and merged into the GNU Radio repository, it cannot be withdrawn or removed by the original author. This ensures that the integrity and continuity of the project's codebase are preserved.

### License Terms

GNU Radio intends to maintain the existing license terms under which contributions are made. We do not intend to change the licensing terms of any contributions after submission. Any potential changes to the license, such as re-licensing, would require an explicit, agreed-upon process involving the contributor and the project maintainers.  Initial license terms are specified in the LICENSE file of the submodule, or in the SPDX header of the individual file if not consistent with the overall submodule.

By submitting a contribution, you agree to the terms of the Developer Certificate of Origin (DCO), which certifies that your contribution is your original work and that you have the right to submit it under the license terms of the specific module.

We accept contributions for in-tree code with the following license preference:

Core: MIT required

Block Library: MIT preferred, with LGPLv3 as an alternative

### DCO Signing 

Code submitters must have the authority to merge that code into the public GNU Radio codebase.
In some cases, the rights to exploit the code may belong to the contributor's employer, depending on jurisdiction
and employment agreements.

For that purpose, we use the [Developer's Certificate of Origin](DCO.txt). It is the same document used by other
projects.
Signing the DCO states that there are no legal reasons to not merge your code.

To sign the DCO, suffix your git commits with a "Signed-off-by" line. When using the command line,
you can use `git commit -s` to automatically add this line. If there were multiple authors of the code, or other types
of stakeholders, make sure that all are listed, each with a separate Signed-off-by line.

#### License Philosophy

GNU Radio 4 uses the MIT License for the core runtime and libraries, with the option for submodules to be licensed under LGPLv3 with a linking exception. This licensing model in combination with the DCO was chosen to support the following goals:

- Maximize Adoption and Enable Public-Private Collaboration: The MIT License reduces legal and logistical friction for development partners - including those with cautious legal teams or incompatible licensing needs - making it easier for academia, industry, and government to integrate, contribute to, and build on GNU Radio. This fosters innovation, accelerates adoption across sectors, and ensures that the broader community benefits from shared advancements.

- Encourage Contributions from a Diverse Ecosystem: By lowering legal barriers, we aim to attract contributors from companies, academic institutions, and individuals who might otherwise avoid contributing to more restrictively licensed codebases due to internal policies or licensing constraints.

- Remain free/libre in the spirit of the open source principles: For certain submodules that implement signal processing algorithms or higher-level blocks, the LGPLv3 license withcan be used to preserve the copyleft spirit of GNU Radio.

- Empower Submodule Authors: We recognize that some contributors may wish to enforce stronger copyleft guarantees. By allowing submodules to choose LGPLv3 + linking exception (and out of tree authors to choose GPLv3 or any other license), we provide flexibility for authors to assert more control over how their code is reused.

- Stay Compliant with Evolving Legal Landscapes: A modular, permissive licensing approach ensures GNU Radio can remain compliant under changing national and international laws - particularly around cybersecurity, product liability, AI governance, and data protection regulations like GDPR.


## Code of Conduct

To ensure an inclusive community, contributors and users in the GNU Radio community should follow
the [code of conduct](./CODE_OF_CONDUCT.md).
