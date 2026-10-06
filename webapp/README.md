# mineralML web app

Browser front end for mineralML: upload a CSV/Excel file of oxide wt% (or type analyses manually, for those so inclined), get classifications with prediction scores, and download a Excel workbook with one sheet per mineral plus stoichiometry sheets. Classification diagrams (TAS, feldspar and pyroxene ternaries, the pyroxene quadrilateral, amphibole, Fe–Ti oxide and spinel), plus custom ternaries and Harker plots, download as PDF, SVG or PNG. No Python needed for users :-).

`streamlit_app.py` is the page; `diagrams.py` builds the composition diagrams (fields from the mineralML classifiers, points redrawn so they can be colored by any column).

## Run locally

```
pip install -r webapp/requirements.txt
streamlit run webapp/streamlit_app.py
```

## Deploy (free)

**Streamlit Community Cloud** (simplest)
1. Push directory to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub and create app.
3. Repository `sarahshi/mineralML`, branch `main`, main file path `webapp/streamlit_app.py`.
4. Optionally set a custom subdomain (e.g. `mineralml.streamlit.app`). Deploy.

The app redeploys on every push. It sleeps after 12 hours without visitors; the next visitor clicks a wake-up button and waits ~30 s.

## Updating the model

The app installs mineralML from PyPI (pinned in `requirements.txt`). After a new release, bump the version pin and push.
