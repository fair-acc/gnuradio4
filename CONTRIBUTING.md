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

GNU Radio does not claim ownership of any contributions you make. All copyrights remain with the original author of the contribution. As the author, you are responsible for maintaining and marking your copyright in the appropriate locations.

When modifying or adding new files, you must include a copyright header in each file you touch. The copyright header should contain the following information:

```
/*
 * Copyright (C) [Year] [Name or Pseudonym of Author]
 *
 * SPDX-License-Identifier: [License of module or file, default LGPL-3.0 for core]
 */
```

If you are modifying an existing file, add your copyright notice below any existing copyright lines. Please use the name you would like to associate with your contribution and ensure consistency across all files you contribute to.  

#### Non-Revocability of Contributions

By submitting a contribution to the GNU Radio project, you agree that your contribution is non-revocable. Once a contribution is accepted and merged into the GNU Radio repository, it cannot be withdrawn or removed by the original author. This ensures that the integrity and continuity of the project's codebase are preserved.

#### License Terms

GNU Radio intends to maintain the existing license terms under which contributions are made. We do not intend to change the licensing terms of any contributions after submission. Any potential changes to the license, such as re-licensing, would require an explicit, agreed-upon process involving the contributor and the project maintainers.

By submitting a contribution, you agree to the terms of the Developer Certificate of Origin (DCO), which certifies that your contribution is your original work and that you have the right to submit it under the license terms of the specific module.

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

#### Notably, by contributing to GNU Radio:

1. You grant this project a non-exclusive, irrevocable, worldwide, royalty-free, sublicensable, transferable license
   under all of your relevant intellectual property rights (including copyright, patent, and any other rights), to use,
   copy, prepare derivative works of, distribute, and publicly perform and display the contributions.
2. You confirm that you are able to grant us these rights. You represent that you are legally entitled to grant the
   above license. If Your employer has rights to intellectual property that You create, You represent that You have
   received permission to make the Contributions on behalf of that employer, or that Your employer has waived such
   rights for the Contributions.
3. You represent that the Contributions are Your original works of authorship, and to Your knowledge, no other person
   claims, or has the right to claim, any right in any invention or patent related to the Contributions.
   You also represent that You are not legally obligated, whether by entering into an agreement or otherwise, in any way
   that conflicts with the terms of this license.
4. We acknowledge that, except as explicitly described in this Agreement, any Contribution which you provide is on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT
   LIMITATION,
   ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

This above is to ensure that the GNU Radio Project:

* Remains free/libre in the spirit of the open source licenses and principles.
* Stays or can be made compliant under international and national laws if these change (notably U.S. and EU stances on
  cybersecurity, product liability, GDPR, and use of AI).
* Encourages public-private/industry partnerships to foster innovation and collaboration, ensuring that all can benefit
  from and contribute to the project.

## Code of Conduct

To ensure an inclusive community, contributors and users in the GNU Radio community should follow
the [code of conduct](./CODE_OF_CONDUCT.md).
