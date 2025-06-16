# %%

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

from .constants import OXIDE_MASSES, OXYGEN_NUMBERS, CATION_NUMBERS, OXIDE_TO_CATION_MAP, CATION_TO_OXIDE_MAP, VALENCES

# %%


class SolidSolutionGenerator:
    # Oxide masses and other constants

    def __init__(
        self,
        endmembers,
        oxygen_basis,
        minor_elements=None,
        site_variation=0.05,
        element_noise_scale=0.1,
        min_site_fraction=0.05,
        mixing_dist="beta",
        mixing_params={"a": 2, "b": 2},
        validate_fn=(None),
    ):
        self.OXIDE_MASSES = OXIDE_MASSES
        self.OXYGEN_NUMBERS = OXYGEN_NUMBERS
        self.CATION_NUMBERS = CATION_NUMBERS
        self.OXIDE_TO_CATION_MAP = OXIDE_TO_CATION_MAP
        self.CATION_TO_OXIDE_MAP = CATION_TO_OXIDE_MAP
        self.VALENCES = VALENCES

        self.endmembers = endmembers
        self.oxygen_basis = oxygen_basis
        self.minor_elements = minor_elements or {}
        self.site_variation = site_variation
        self.element_noise_scale = element_noise_scale
        self.min_site_fraction = min_site_fraction
        self.mixing_dist = mixing_dist
        self.mixing_params = mixing_params
        self.validate_fn = validate_fn or (lambda x: True)
        self.suffix = f"_cat_{self.oxygen_basis}ox"
        self.element_to_oxide = self.CATION_TO_OXIDE_MAP

        # Validate oxygen counts in endmembers
        self._validate_endmember_oxygens()

    def _validate_endmember_oxygens(self):
        """Check all endmembers have the specified oxygen basis."""
        for name, cations in self.endmembers.items():
            if cations.get("O", self.oxygen_basis) != self.oxygen_basis:
                raise ValueError(
                    f"Endmember {name} has {cations.get('O')} oxygens "
                    f"but should have {self.oxygen_basis}"
                )

    def _generate_mixing_fraction(self):
        """Generate mixing fraction based on specified distribution."""
        if self.mixing_dist == "beta":
            a = self.mixing_params.get("a", 2)
            b = self.mixing_params.get("b", 2)
            return np.random.beta(a, b)
        elif self.mixing_dist == "uniform":
            return np.random.uniform(0, 1)
        elif self.mixing_dist == "dirichlet":
            alpha = self.mixing_params.get("alpha", [1] * len(self.endmembers))
            return np.random.dirichlet(alpha)[0]
        else:
            raise ValueError(f"Unknown mixing distribution: {self.mixing_dist}")

    def _add_minor_elements(self, cations):
        """Add minor elements to the composition."""
        for element, params in self.minor_elements.items():
            if params["distribution"] == "exponential":
                scale = params.get("scale", 0.01)
                max_frac = params.get("max_fraction", 0.02)
                amount = np.random.exponential(scale)
                amount = min(amount, max_frac * sum(cations.values()))

                # Reduce major elements proportionally
                total_majors = sum(cations.values())
                for maj in cations:
                    cations[maj] *= 1 - amount / total_majors
                cations[element] = amount

        return cations

    def _apply_site_variation(self, site_totals):
        """Apply variation to site totals using all noise parameters."""
        varied_totals = {}

        for site, tot in site_totals.items():
            # Get site-specific variation if specified, else use global
            variation = (
                self.site_variation.get(site)
                if isinstance(self.site_variation, dict)
                else self.site_variation
            )

            # Base lognormal variation (preserves positives)
            base_variation = tot * np.random.lognormal(mean=0, sigma=variation)

            # Add additional Gaussian noise scaled by element_noise_scale
            gauss_noise = np.random.normal(loc=0, scale=self.element_noise_scale * tot)

            # Combine all noise sources
            varied_total = base_variation + gauss_noise

            # Enforce minimum site fraction using min_site_fraction parameter
            varied_totals[site] = max(
                varied_total,
                self.min_site_fraction * tot,  # Now using the parameter
            )

        return varied_totals

    def _add_element_noise(self, cations):
        """Add per-element noise using element_noise_scale parameter."""
        noisy_cations = {}
        total_charge = 0

        # First pass: add noise to each element using element_noise_scale
        for element, count in cations.items():
            if element == "O":
                continue

            # Use the instance's element_noise_scale parameter
            noise_scale = self.element_noise_scale * count
            noise = np.random.normal(loc=0, scale=noise_scale)
            noisy_count = max(count + noise, 0)  # Don't go negative

            noisy_cations[element] = noisy_count

            # Track charge (simple valence model)
            valence = self._get_element_valence(element)
            total_charge += noisy_count * valence

        # Second pass: adjust to maintain approximate charge balance
        target_charge = 2 * self.oxygen_basis  # O²⁻ charge
        charge_ratio = target_charge / total_charge if total_charge != 0 else 1

        # Adjust cations proportionally
        for element in noisy_cations:
            if element != "O":
                noisy_cations[element] *= charge_ratio

        # Perform oxygen-basis renormalization
        current_total_oxygens = sum(
            noisy_cations[elt] * self.OXYGEN_NUMBERS[self.element_to_oxide[elt]]
            for elt in noisy_cations
        )
        scale_factor = self.oxygen_basis / current_total_oxygens
        for elt in noisy_cations:
            noisy_cations[elt] *= scale_factor

        return noisy_cations

    def _get_element_valence(self, element):
        """Valence dictionary lookup for common elements."""
        return VALENCES.get(element, 2)  # Default to 2 if unknown

    def _total_charge(self, cations):
        """
        Compute charge (cation_count*valence), stripping "_cat_{N}ox" suffix.
        """
        total = 0.0
        suf = self.suffix  # e.g. "_cat_{N}ox"
        for key, cnt in cations.items():
            # If the key ends with "_cat_{N}ox", remove that part:
            if key.endswith(suf):
                elt = key[: -len(suf)]
                print(elt)
            else:
                elt = key

            val = self._get_element_valence(elt)
            total += cnt * val

        return total

    def _check_charge_balance_add_noise(self, cations): 
        """Check charge and add some noise."""

        raw_charge = self._total_charge(cations)
        expected_charge = 2 * self.oxygen_basis
        if abs(raw_charge - expected_charge) > 0.2:
            print(f"Charge mismatch: {raw_charge:.2f} vs {expected_charge}")
        return self._add_element_noise(cations)

    def _calculate_oxide_wt_percent(self, cations):
        """Convert cation counts to oxide weight percentages."""
        oxide_wt = {}
        total_mass = 0

        for element, cat in cations.items():
            if element == "O":
                continue

            oxide = self.CATION_TO_OXIDE_MAP.get(element)
            if not oxide:
                raise ValueError(f"No oxide mapping for element {element}")

            per = self.CATION_NUMBERS[oxide]
            moles = cat / per
            mass = moles * self.OXIDE_MASSES[oxide]
            oxide_wt[oxide] = mass
            total_mass += mass

        # Normalize to 100%
        return {ox: (mass / total_mass) * 100 for ox, mass in oxide_wt.items()}

    def generate(self, n_samples=1000):
        """Generate synthetic mineral compositions."""
        records = []
        endmember_names = list(self.endmembers.keys())

        for _ in range(n_samples):
            # 1) Endmember mixing
            if len(self.endmembers) > 1:
                frac = self._generate_mixing_fraction()
                cations = {}

                # For binary mixtures
                if len(endmember_names) == 2:
                    for element in set().union(
                        *[e.keys() for e in self.endmembers.values()]
                    ):
                        if element == "O":
                            continue
                        cations[element] = frac * self.endmembers[
                            endmember_names[1]
                        ].get(element, 0) + (1 - frac) * self.endmembers[
                            endmember_names[0]
                        ].get(element, 0)

                    # check raw charge before noise
                    cations = self._check_charge_balance_add_noise(cations)
                else:
                    # For >2 endmembers (requires dirichlet mixing)
                    for i, name in enumerate(endmember_names):
                        for element, value in self.endmembers[name].items():
                            if element == "O":
                                continue
                            if element not in cations:
                                cations[element] = 0
                            cations[element] += frac[i] * value

                    # check raw charge before noise
                    cations = self._check_charge_balance_add_noise(cations)

            # 2) Coupled site generation (alternative)
            else:
                only = next(iter(self.endmembers))
                # copy its cations (drop oxygen)
                base_cat = {
                    el: cnt for el, cnt in self.endmembers[only].items() if el != "O"
                }

                # apply log-normal variation to the total cation sum
                total = sum(base_cat.values())
                varied_total = total * np.random.lognormal(
                    mean=0, sigma=self.site_variation
                )
                scale = varied_total / total
                cations = {el: cnt * scale for el, cnt in base_cat.items()}
                # now add per‐element Gaussian noise, renormalize charge & O
                cations = self._check_charge_balance_add_noise(cations)

            # Add minor elements
            cations = self._add_minor_elements(cations)

            # Convert to oxide wt%
            oxide_wt = self._calculate_oxide_wt_percent(cations)

            # Validate
            if not self.validate_fn(oxide_wt):
                continue

            # Add cation counts (normalized to oxygen basis)
            norm_cations = {
                f"{k}_cat_{self.oxygen_basis}ox": v
                for k, v in cations.items()
                if k != "O"
            }
            records.append({**oxide_wt, **norm_cations})

        return pd.DataFrame(records)

    def compare_distributions(
        self,
        base_df,
        synth_df=None,
        n_samples=1000,
        ncols=3,
        figsize_per=(4, 3),
        suptitle=None,
    ):
        """
        Compare cation-count distributions between base_df and synth_df.
        """
        if synth_df is None:
            synth_df = self.generate(n_samples=n_samples)

        suffix = self.suffix
        valid = []
        for bcol in base_df.columns:
            if not bcol.endswith(suffix):
                continue
            el = bcol[: -len(suffix)]
            scol = f"{el}{suffix}"
            if scol in synth_df.columns:
                valid.append((el, bcol, scol))

        if not valid:
            print("No matching cation columns.")
            return pd.DataFrame()

        stats = []
        nrows = -(-len(valid) // ncols)  # Ceiling division
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * figsize_per[0], nrows * figsize_per[1]),
            squeeze=False,
        )
        axes = axes.flatten()

        for ax, (el, bcol, scol) in zip(axes, valid):
            # Get cation data
            base_c = base_df[bcol].dropna().values
            synth_c = synth_df[scol].dropna().values
            ks = ks_2samp(base_c, synth_c)

            stats.append(
                {
                    "cation": bcol,
                    "ks_stat": ks.statistic,
                    "p_value": ks.pvalue,
                    "mean_base": base_c.mean(),
                    "mean_synth": synth_c.mean(),
                    "std_base": base_c.std(ddof=1),
                    "std_synth": synth_c.std(ddof=1),
                }
            )

            # Plot cation distributions
            parts = ax.violinplot(
                [base_c, synth_c], positions=[1, 2], showmeans=True, showmedians=False
            )

            # Style the violins
            for pc in parts["bodies"]:
                pc.set_facecolor("#0C7BDC")
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)

            parts["cmeans"].set_color("red")
            parts["cmins"].set_color("black")
            parts["cmaxes"].set_color("black")
            parts["cbars"].set_color("black")

            ax.set_ylabel("Cation Count", fontsize=10)
            ax.tick_params(axis="both", which="major", labelsize=8)

            # Plot oxide wt% on twin axis if available
            ox = self.element_to_oxide.get(el)
            if ox and ox in synth_df.columns:
                base_w = base_df[ox].dropna().values
                synth_w = synth_df[ox].dropna().values

                ax2 = ax.twinx()
                parts2 = ax2.violinplot(
                    [base_w, synth_w],
                    positions=[4, 5],
                    showmeans=True,
                    showmedians=False,
                )

                # Style the second set of violins
                for pc in parts2["bodies"]:
                    pc.set_facecolor("#FF7F0E")
                    pc.set_edgecolor("black")
                    pc.set_alpha(0.7)

                parts2["cmeans"].set_color("red")
                parts2["cmins"].set_color("black")
                parts2["cmaxes"].set_color("black")
                parts2["cbars"].set_color("black")

                ax2.set_ylabel("Oxide wt%", fontsize=10)
                ax.set_xticks([1, 2, 4, 5])
                ax.set_xticklabels(
                    [
                        "Natural\nCations",
                        "Synthetic\nCations",
                        "Natural\nOxide",
                        "Synthetic\nOxide",
                    ],
                    fontsize=8,
                )
            else:
                ax.set_xticks([1, 2])
                ax.set_xticklabels(
                    ["Natural\nCations", "Synthetic\nCations"], fontsize=8
                )
            ax.set_title(el, fontsize=12, pad=10)

        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=1.02)
        for extra_ax in axes[len(valid) :]:
            extra_ax.axis("off")
        plt.tight_layout()
        plt.show()

        return pd.DataFrame(stats).set_index("cation")


# %%
