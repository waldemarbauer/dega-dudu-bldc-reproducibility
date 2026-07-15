"""Generate manuscript result figures and publication assets from persisted outputs."""

from __future__ import annotations

from _stage_runner import run_script


if __name__ == "__main__":
    # ArticleV1 intermediate scientific figure data used by the final M3 set.
    run_script("Scripts/Publication/generate_article_v1_figures.py")
    run_script("Scripts/Publication/generate_article_v1_xai_figures.py")
    run_script("Scripts/Publication/build_article_v1_publication_package.py")
    # Manuscript-aligned computational figures: Figures 6–14 in the attached version.
    run_script("figures_m3/scripts/make_all_figures.py")
    run_script("figures_m3/scripts/make_captions.py")
    run_script("figures_m3/scripts/validate_m3_figures.py")
