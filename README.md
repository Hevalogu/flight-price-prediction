# Flight Price Prediction

Predicting Indian domestic flight prices from airline, route, timing, stops, class and days left before departure, and answering plain-English questions about prices through a small local AI agent.

**Model result:** A Random Forest predicts ticket prices with an average error of **₹1,090 (R² 0.985)**, about **4× better** than a Linear Regression baseline (₹4,501). On Economy tickets, the error is **11%** of the median fare, down from **48%**.

**Agent result:** A tool-calling agent on a local 3B model (Qwen 2.5 via Ollama) answers questions such as *"cheapest flight from Chennai to Bangalore on the 25th"* using the trained model for every price. It passes **7/7** regression tests, including refusing requests the data can't support.

---

## Data

[Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) by Shubham Bathwal on Kaggle: 300,153 tickets between six Indian metro cities (Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai).

The first notebook downloads the data automatically with `kagglehub`.

## Project structure

```
flight-price-prediction/
├── Notebooks/
│   ├── 01_data_exploration.ipynb   # loading, health checks, cleaning, EDA
│   └── 02_modelling.ipynb          # encoding, models, evaluation, question answering, agent
├── Data/                           # created by notebook 01 (not tracked in Git)
├── Models/                         # trained model, created by notebook 02 (not tracked in Git, ~86 MB)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
```

The question-answering and agent sections also need [Ollama](https://ollama.com) running locally, with the model downloaded:

```bash
ollama pull qwen2.5:3b
```

Then run the notebooks **in order**:

1. `01_data_exploration.ipynb` downloads the data, cleans it and saves `Data/flights_clean.csv`
2. `02_modelling.ipynb` trains the models, saves the Random Forest to `Models/`, and builds the question-answering layer and the agent

## Part 1: Price model

### Approach

1. **Health checks:** no missing values and no duplicate rows. Dropped `Unnamed: 0` (a row counter) and `flight` (1,561 codes acting as IDs; the useful part is already in `airline`).
2. **Exploratory analysis:** looked at how each feature relates to price, always splitting by class, because Business and Economy fares differ so much.
3. **Encoding:** `stops` as an ordered number (0/1/2) because price rose steadily with each stop; `class` as 0/1; airline, cities and times as one-hot columns (first category dropped).
4. **Train/test split:** 80/20, `random_state=42`.
5. **Models:** Linear Regression as a baseline, then Random Forest (50 trees).
6. **Evaluation:** MAE, RMSE and R², overall and **split by class**.

### Key findings from exploration

| Feature | Effect on price |
|---|---|
| Class | Business median ₹53,164 vs Economy ₹5,772 (about 9×) |
| Days left | Economy fares jump sharply about 15 days before departure; day-1 fares are about 3× those booked 6–7 weeks ahead |
| Airline | Only Vistara and Air India sell Business. In Economy, AirAsia is cheapest and Vistara the most expensive (about 2×) |
| Stops | More stops goes with a higher price in both classes, contrary to my expectation |
| Duration | One-stop flights take about 5× as long as non-stop (12.25 vs 2.25 hours), mostly layover |
| Route | About 1.5× between cheapest and dearest Economy routes; price does not follow distance |
| Departure / arrival time | Weak effect; arrival time matters more than departure, mainly in Business |

### Results

| Model | MAE | RMSE | R² | Negative predictions |
|---|---|---|---|---|
| Linear Regression | ₹4,501 | ₹6,815 | 0.910 | 2,629 |
| **Random Forest** | **₹1,090** | **₹2,811** | **0.985** | **0** |

**Error by class (MAE as % of median fare):**

| Class | Linear Regression | Random Forest |
|---|---|---|
| Economy | 48% | 11% |
| Business | 16% | 4% |

**Why the linear model fell short.** Its R² of 0.910 looks strong, but it comes mostly from separating Economy and Business. Split by class, Economy errors were about half the fare, and every negative prediction was an Economy ticket. A linear model has one weight per feature, shared by both classes, so it can't learn that a stop adds about ₹2,400 in Economy but about ₹26,700 in Business, or the stepped days-left pattern. Decision trees branch on class first and learn separate rules for each, and their predictions are averages of real prices, so they can't go negative.

**What the Random Forest relied on.** `class` dominates feature importance (0.880), followed by `duration` (0.059) and `days_left` (0.019). `stops` doesn't reach the top 10, likely because `duration` already carries that information.

## Part 2: Asking questions in plain English

The model only understands 29 numeric columns, so a layer on top turns a question into those inputs:

1. **`predict_price()`** fills in the 29-column format for one flight. Sanity check: Vistara, Delhi → Mumbai, Economy, non-stop, 5 days out predicts ₹6,082 vs a median of ₹5,955 across 15 real tickets.
2. **`quote_price()`** fills in details the person didn't give (airline, stops, time slots) with the most common option on that route, and the reply always states these assumptions.
3. **Reading the question**, in two versions:
   - **Rules** (`parse_question`): fast and exact for phrasing they know ("in two weeks", "Friday", "Bombay"), useless for anything else
   - **Local LLM with guardrails** (`llm_parse_question`): understands more phrasing (e.g. "on the 25th"), while rules count dates and overrule anything the question never mentioned. Example: the LLM turned "Madras" into Mumbai; the city guardrail corrected it to Chennai.

## Part 3: The agent

Instead of fixed steps, the LLM **decides which tool to call**, reads the result, and repeats until it can answer.

```
Question → LLM chooses a tool → Python runs it → result goes back to the LLM → ... → answer
```

**Tools**

| Tool | Job |
|---|---|
| `get_price` | Price one flight, using the Random Forest |
| `compare_options` | Price every airline and stops option on a route, return the cheapest or most expensive |
| `days_until` | Turn a calendar date or weekday into days from today |

**Design principle: the LLM handles language, the model handles numbers, and Python checks the LLM.** With a 3B model, reliability came from guardrails and well-designed tools rather than from the model itself:

- **Python decides any input it can read reliably** from the question (number of days, cities, class, cheapest vs most expensive). The LLM's values are only used where Python finds nothing, such as a misspelled "banglore".
- **Complex steps live inside tools.** Asking the 3B model to list 14 options and price each one failed; a single `compare_options` tool fixed it.
- **The loop sends the agent back** if it skips a required step (e.g. answers "cheapest" without comparing options), at most twice, with a step limit.
- **Every ₹ amount in the final answer must appear in a tool result**; otherwise the tool's own result is returned instead.

**Example**

> **Q:** i want to know the cheapest flight from chennai to banglore on 25th Nov
> **A:** The cheapest flight from Chennai to Bangalore on the 25th of November is ₹3,361 with SpiceJet, a non-stop flight in Economy class.

Before the guardrails, the same question was priced 25 days ahead instead of 7, and labelled "cheapest" without comparing anything: a confident, wrong ₹5,569.

### Regression tests

The notebook ends with automated tests. Expected prices were verified directly against the tools; for impossible requests the test checks that no price is invented.

| Question | Expected | Result |
|---|---|---|
| Delhi → Mumbai in 5 days | ₹9,062 | ✅ |
| Cheapest Chennai → Hyderabad in 14 days | ₹4,204 | ✅ |
| Most expensive Delhi → Kolkata in 10 days | ₹18,801 | ✅ |
| Cheapest Chennai → "banglore" on 25th Nov | worked out from the tool on the day | ✅ |
| Business class on IndiGo | no price (IndiGo sells no Business) | ✅ |
| Delhi → Goa | no price (Goa not in the data) | ✅ |
| Delhi → Mumbai, no date | asks for the date | ✅ |

The first full run passed only 3 of 7, even though each question had looked fine when tried on its own. Fixing what the tests exposed took it to 6/7, then 7/7.

## Limitations

- **Model:** train MAE (₹425) is lower than test MAE (₹1,090): the forest fits its training data closely, though it still generalises well. Only 0.74% of tickets are over 30 hours long, so very long journeys are less reliable. The data covers six metro cities and 1–49 days ahead only.
- **Defaults matter:** when airline or stops aren't given, the most common option on the route is assumed, which can double the price compared with a non-stop flight. Replies state these assumptions.
- **Agent:** the 3B model often calls `get_price` first and needs a guardrail reminder to compare options. For "is ₹9,000 a good price?" it calls the right tool but reasons weakly about the comparison.
- **Dates:** `days_until` uses the day number only, not the month, so "25th Nov" means "the next 25th". Vague phrases such as "end of next week" are read as 7 days.
- **Final wording** is written by the LLM and can be loose (e.g. "two stops" for "two or more").

## Next steps

- A price-range tool so the agent can judge whether a given price is good
- Tune the Random Forest (e.g. `min_samples_leaf`) and use cross-validation; try gradient boosting
- Try a larger local model and measure how many guardrails it still needs
- A small web app (e.g. Streamlit) for the chat
