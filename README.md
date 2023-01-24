[![License](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://opensource.org/licenses/LGPL-3.0)
![CMake](https://github.com/fair-acc/graph-prototype/workflows/CMake/badge.svg)
# Graph Prototype
A small proof-of-concept for evaluating efficient [directed graph](https://en.wikipedia.org/wiki/Directed_graph)-based
algorithms, notably required node structures, scheduling interfaces, and partial compile-time merging of
[directed acyclic](https://en.wikipedia.org/wiki/Directed_acyclic_graph) as well as 
[cyclic graphs](https://en.wikipedia.org/wiki/Feedback_arc_set) (aka. feedback loops).  

The expressed goal is to guide the low-level API design and functionality for the upcoming 
[GNU Radio 4.0](https://github.com/gnuradio/gnuradio/tree/dev-4.0) release.

A [single header version](https://raw.githubusercontent.com/fair-acc/graph-prototype/single-header/singleheader/graph.hpp) is provided in the `dist/`subdirectory, which can be regenerated with the merge_headers.sh script.
It can be used on [compiler-explorer](https://godbolt.org/) with `-sdt=c++20 -O3` compiler options: [compiler-explorer example](https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQWm8GA6ZAB36kqqBgQDKyJvRABGACzDRBAKoBnTAAUAHiADkABj2kAViABMpWkwahkAUnMAhR09Ib6yAnlGVG6AGFUWgBXFgYLS38AGW5MADkwgCNMYgtSflQ1Qh8GINDwyIysnNFYhgTk1PSPTC9c8QImYgJ8sIjzS1r60UbmgnLKlhS0rqaWtsLO93GBuMTh6vMASndUEOJkdj1HAGZuZFCsAGp7XYDm4iYATzPsewMAQT2Do8xT86oWAgB6NGJMDwEHcHs8ni8GIcQidHOYEAQCPw1CAfj8rgB3HjAQgIEJJEIaTbKRgEPioFg/KhcYgAWiYyGQP2AV34CBp/GIqCIBGu/EwP2ytnoNIQmCYWGIAu4wHoovFqSZLIQQMEsNBoOYbDU/Hp7yowA%2BABFjlS8GkQMymKyzi5wU8CJgWPxrA6zgEeXzNe8ACogp6HJhqNTHK4MdDkgD6anWm3eIGO/DxtDwyBNoBADAwmDdofDLCjMa2bt9u2wpDTIBAAHkVCWAt7y7Dc%2BS1aW7qcAOy2x6JpLJ5AgUHHYfHNAMNQO7Qc46%2Bp4jhOcrZBiOiTAQZad7vz%2BcAggbBjHeTmG1Dkf2DuG9UXk92x4Op0u7PnD2MVg%2Bv2PANB0frAhR7gANbHPGvb9hWGZZm6aAhH%2BgoAcWdzlvqlYAJLxMWjadHBrb3KWm6gqBKaDnOI4AG6oHg6CnsOHKoEuagrhUEDescpHSCEmAbueW7bsOE7oJW0EEB8ARuqx7HvG6Yn8ZW/i0Dejzzuel7gteuzdqC97OkwrrPryr5sDO5YsZI0iYOWL5escABKRrHFghwvhAMkgA5tBsfJ5wltg67HAAVMcLluR5CGluuywfl%2BwZqFI9DAQmSYpuBmZYG6MVmRhxymfQ5bWYh4FoZlsIkHg2LMPJnT5ch1a1m61mYeY6X0FRlVtnh3EEYlA7UccWmPm6prmmwTSVn%2BJD/iwVFee2ABqH5KQArC4S0pXg6XEFRC2XltP7jpO05MDBqA9bR9GMWuM3HEwG5jhOxyZpg2hbPwwkdSRvG7vuV3%2BVlsVPjxw7KVel7qeq9qOtpunuvpVkNr1MNvjZdluU5QV1O50ihT5XHOIFBACa56MhdNYXLBFbb%2BtY37ihK8WEam1UpU%2BFzoBKWNIemhVeQ1NP%2BAYOEc6h6Hc6cnS82GsgC%2BBNZ1vVouNWEOHtm9PZdcRikjn1OnM4NlbDUwo0rsQE1TfWs3zWeS32CtGBrVIG3W9txq3ftxBXUdJ2Lpgy6rhAl1MOWl1JDdoh3Q9T2YC9%2BHvdun3EAeTCi04xxJApSnXqpIMaeDD7a26lmI3DLnZAAXpgEbCUEIRiHZx4U5%2BVPBugITOim2t011yWQeczet1IUNw1XYhVdSeuYCNIAvsmE4DZzwv1g13BK6WgsgACfLa%2BgEaZC0ahukPBDGQ1v6wqvO9/i%2BFeVuI3rWdgjwALJnyQf74ACPQMFfNXepotVtdg0cNbDgJNKFOgZJK7GNIzbuARe79jziLA%2BI8zRjwnlPNaBBZ5CyKp0JerVcJlnAuvMUDot7nz3ucA%2BR95Yn2mOBc%2BFd9JfxvnfR%2Bz8WgRjfnUbwogv4y1/t5EEoNbz03VvOEBtgQzj33IwvkdkC6GSSOAyscdP4vgUj1a2y0nCrXWptJ2u0JyPWnKo2R2YY6nW9gxX2LFrqGOEuHZ6r0uw9R3NI%2BOpwrbmAAGxMEdmlfGlZS7l1JDwHgxwUIUJ8i5bgWBtBRkwAARw4pCZmkSwk8DuFxLsUi9weJcpwACISW70CgBAcilENyRPLNdDJ5Nk5AxjvOZygSOBMCKRGWJj0EnJMYEWSh6xh6k3qZojOzw1LZzvBDfqelPSF3LMXPAZcK7HENJHAgwJ65uOSWab2EA1kvQQB8ABBgIqU0DE3TA1hrgdz7ElaBqUe5XJuJlA5GyqpzxweYPBddCHVRll82hvzlYuJji5S4LyRZvM2QA/EVAqCpDsh1CZPVuDCV4hi%2BcmRgxnkgccfmwiwSqzud1GOWitG6Ptvox29jjFu1nEAhcdErHnWYscbgG5HHrMAZiliu4MVnGNHChFxAtHYv8YSzFwrUhiqyDSwV7KGBpwxXgKgxwIDYqNAq6FxwaTHFkNkgGmKExZDsgSo16djRXI0Dy41gNnCuHFZKgVYzMWqNycqwGYygaEtBGi0EnBuC%2BRVuIwUBpqpsGIMAf6PUJHhvTJG6NEYkjXE6WGR6GiyWgsZfOVE9l0bND1LQVA6IaSWlZPGcwxxtCKt1e2XmbsaTtiapgGkKaaRVqbXhFtbbrg0hYNwAksg614V/ImTBTSRyHSIMcRNmB0B2Tncm1NXTtBugMOWAl0S/pujReWGkktSbliXSmtNcT12bruM5Hd5w93HF%2BeuGprMZW3qGdjMmmac0jjzY9OoMF3jluBJOvirSIW3FfYfQ89akU5NkOWSwxxdjlnkJ2LOriRzguIFccDAQ73yHbEkGDydZAbv1aRkj5YSOoc/ZitFdqPpms9duEQbsWkE2CSsvAjH1LspElBnjrhXB4ENehhjgm8VzvQDwSxPsmJ%2BKWngR25ZU4KcdiM4D1HbyYq%2BAQSsHI0UQFhNZb2IRaDCVQGqwDxxf3IBgrkeMyLtoBCVXQ4g6miWYouakAgEA3aCoVeYU5THGkebPNmzzR1Z2pGjQuhVJ6V3prXecUjW6IBwLblDO9D7ViBRvbhsQ8Gr0ft9Rpn92g/0OmOIB0TeMCZgd3QV/jADE4Ku4vq%2BD5YkOHmoyVr9w4WPqsWcs4SXGFXmt42JfDAmHXOGE7au107UC9RKe8XFxpJPSa9rJtc8mnCKa2u5%2Bj7sZ1aOIHBkMx4doKr3M6GNGn5w6b08QAzaoXEXnwu962znT4hnO8QFYTHLWjJUqFr1Fqp2Rck4u6LzNYQtpagh2E4t0CS3wdejKEHCtHqus%2B0VEGivuZq2VirAGlQ1cw9hhrkGpvNaI%2B1%2B9nXkM9fByBurWHIX5epwRunFGyOUfIwYZnYM%2BuKvRUd4cfm8XmpqwNtjQSlkhIm1Lm0SuAj8YaTNvbIn7sjj8w6iTMOpMyesXJrR%2B3DTKbN2pwHYWQcy%2B%2BE9l7nRjNqFM%2BZyzSprPldszwiIH2nMuYQ25mjGKvMtF81qvF3igvOq9XbrNLPjtLah3FmHy6z0ZvOOd1L8WM9JYCClq96X%2B7Mzw0VhZeWssE4rxjznWP30h94sTn3pOrRAZFxTjnZe8ItbxW187CGusoZ9ZM7TJBButI4yN7jydRvnHV0nIT2uRfzkW8t27R3U9RvnZt5l22IC7fN4d%2Bja/Tt/aD114g%2BGrt4pu/QG3vFHsgH02IQzlU3vGkc/77/X3A/ljO//hfv/vIMfrxCFkDlpmFsnGvlgNcvOsjM8jhlXm2B/iriFqMonmvs2CwHZNgQWBsP0pziCKgb1hFjOingbtvunquhevilegUqgKRGuNgTlkJP%2BAwPBPjsMg/lFtvkbltibmuKAZpqDvNtuFgTYHmLgZIZGNGAQaXm%2Bo5o3uIZDobtDtvlJJ0NgT9sVM9mVJjGjgwUwb5jISwDli2lTvXuFDwRtsbqysIegWMnoKsLQPoAtMYBEIYMYKgG0twHwIIKQLpl4WTKQABCAAtBuq4XoPIB4foEYKQD4XoMYMiBukEUYKsHALADAIgCgOSPwHQKkOQJQGgE6AUWkMALINMFgKRCmJgDNHgJgOiFWJ6PoLIMYDQGZqkMiBAEkHEcYEkNwM0NcK0f0YMcQNcFWEkCYNwiMaQCUWwGIFWAwLQMMV4aQFgJwEKNsPEVwvUEwciGsTZv%2Bn0eQGIFciccmEkFcOMUEFgCcQQM9iwCMasFQNYMAGoPUY0c0YwLMSIGINlOwAoEoGIOoFoLoGsWYHBtYLYCgIJs4FYHgEkMiLAF6CADpqQEwWkHoAAByyBtEhGoAvS5AHE0j8SCoOCa6Ba6pVi7C6pUB8jEADpBh4BMHJFXLcL2YQD%2BCTARBwYxDzBVBpBwbYqlB5DBDtByDFDZC%2B6DALAjCSndC%2B59ATDimFBwaKkNCzCymCkKmzA8m6n9DamLBCmrByGxgWAuFuGxFrGJHVrYneI0jeIobAAMj6rmA8BVoQC4CEDj57ArDGBpHOGrBygSiUCrBhERFWD6AxGkBPGRmeHxGJHJEgCpF9EZHZEQBIBjowRFEQAlH5H0DEDxBvj6DaD2mOnOmumVEenGDzo%2BnPYCSWB/ESB/RyCKDNmgk6AnGQmkDogsjPFRl6DuGkAJneH6BVgwTjrHAWZ2kOlOlVZVnumelBClGFmiy7AGoBlpnBliihmPpRExlxkbqjkJH6DJmpnBHhnhGRH6C7DWmJlnmBHbmDnmD3ljlJFPmXkYldH2byBAA%3D%3D)

## Copyright & License
Copyright (C) 2018-2022 FAIR -- Facility for Antiproton & Ion Research, Darmstadt, Germany<br/>
Unless otherwise noted: [SPDX-License-Identifier: LGPL-3.0-or-later](https://spdx.org/licenses/LGPL-3.0-or-later.html)

### Contributors
 * Ivan Čukić <ivan.cukic@kdab.com>
 * Matthias Kretz <M.Kretz@GSI.de>
 * Alexander Krimm, <A.Krimm@GSI.de> 
 * Ralph J. Steinhagen, <R.Steinhagen@GSI.de>

### Acknowledgements
...

