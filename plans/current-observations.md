# EDA

When we plot the raw price we get a skewed graph, when we plot the log price we can clearly see two different largely non overlapping classes, which means there could be a feature that is deciding the split.

After further analysis we find that flight class fits naturally here and The populations can be attributed to Economy and Business class. 

Inside each class, Airline is the second biggest factor -

- **Budget** carriers' Economy costs about ₹7–9k.
- **Premium** carriers' Economy is *always* above ₹20k, about the price of budget Business.

**2. Data quality, which is the 2 marks for preprocessing.**

- Missing `airline` can be filled **exactly** from the flight-code prefix.
- Missing `duration` can be filled for about 98% of rows from other rows of the same flight on the same route.
- Missing values are unrelated to price, so filling them won't bias the model.
- 279 exact duplicate rows have to be dropped *before* splitting into folds, or the CV score comes out too optimistic.

**3. Which features are weak.**

- `days_left` only matters in the last ~4 days, against the plan's expectation.
- Time slots barely matter at all.
- Stops and duration overlap, since more stops means longer flights



## Submission -

The public Submissions are our predictions on the test data. This is just for the public leaderboard and there are hidden test cases as well and they will not be revealed. Hence it's better to use Cross-Validation score as compared to a Loss function on the entire dataset. 
