# How to update my wiki (my-wiki)

# 1) Activate the virtual environment (run these first from any terminal)
cd ~/my-wiki
source .wiki-venv/bin/activate

# 2) Add a new page (example: probability.md)
# Run:
touch docs/probability.md
# Then open and edit docs/probability.md (in VS Code or TextEdit) with content like:
# ----------------------------------------------------
# # Probability
#
# This page is about random variables and distributions.
#
# Inline math: $P(A \mid B) = \frac{P(A \cap B)}{P(B)}$.
#
# Block math:
#
# $$
# \sum_{k=0}^n \binom{n}{k} = 2^n
# $$
# ----------------------------------------------------

# 3) Add the page to the navigation in mkdocs.yml
# Open mkdocs.yml and make sure nav looks something like:
# ----------------------------------------------------
# nav:
#   - Home: index.md
#   - Math:
#       - Algebra: algebra.md
#       - Calculus: calculus.md
#       - Probability: probability.md
#   - My First Note: my-first-note.md
# ----------------------------------------------------
# (Just add the "Probability: probability.md" line under Math.)

# 4) Preview the site locally (optional)
# Run:
mkdocs serve
# Then open http://127.0.0.1:8000/ in a browser.
# Press Ctrl + C in the terminal to stop the server when done.

# 5) Commit and push changes
# Run:
git add docs/probability.md mkdocs.yml
git commit -m "Add Probability page"
git push

# 6) Redeploy the live site
# Run:
mkdocs gh-deploy
# Then open this URL to see the updated site:
# https://pj1911.github.io/my-wiki/
