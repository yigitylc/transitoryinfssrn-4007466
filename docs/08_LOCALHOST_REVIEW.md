# Localhost Review Checklist

Run:

```powershell
streamlit run app/streamlit_app.py
```

Review in the browser:

- [ ] App loads without traceback
- [ ] Data source and sample dates are visible
- [ ] Baseline method is visible
- [ ] Inflation units are understandable
- [ ] Paper replication and live-safe outputs are not mixed
- [ ] TINF 4/8/12 values make intuitive sense
- [ ] Decay/convergence numbers disclose assumptions
- [ ] Any chart can be explained in one sentence
- [ ] No obvious stale placeholder text
- [ ] No secrets or local machine paths displayed

Write observations in `reports/notes/localhost_review.md` before Git push.
