import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  useState,
  type FormEvent,
} from "react";
import ReactMarkdown from "react-markdown";

import { AppLayout } from "../components/AppLayout";
import {
  createPrediction,
  type PredictionRequest,
  type PredictionResponse,
} from "../services/api";

const initialForm: PredictionRequest = {
  age: 35,
  sex: 0,
  bmi: 24,
  high_blood_pressure: 0,
  high_cholesterol: 0,
  diabetes: 0,
  smoking_history: 0,
  current_smoking: 0,
  exercise: 1,
  alcohol_consumption: 888,
  mentally_unhealthy_days: 0,
  fruit_intake: 555,
  fruit_juice_intake: 555,
  other_vegetable_intake: 555,
  green_vegetable_intake: 555,
  time_since_cholesterol_check: 1,
};

function readableFeatureName(
  feature: string
) {
  return feature.replaceAll("_", " ");
}

export function AssessmentPage() {
  const [form, setForm] =
    useState<PredictionRequest>(
      initialForm
    );

  const [result, setResult] =
    useState<PredictionResponse | null>(
      null
    );

  const [submitting, setSubmitting] =
    useState(false);

  const [error, setError] =
    useState("");

  function updateNumber(
    field: keyof PredictionRequest,
    value: string
  ) {
    setForm((current) => ({
      ...current,
      [field]: Number(value),
    }));
  }

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    setSubmitting(true);
    setError("");
    setResult(null);

    try {
      const prediction =
        await createPrediction(form);

      setResult(prediction);

      window.setTimeout(() => {
        document
          .getElementById(
            "prediction-result"
          )
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
      }, 100);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "The prediction could not be generated."
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AppLayout>
      <section className="assessment-header">
        <span className="eyebrow">
          <ShieldCheck size={17} />
          Secure cardiovascular screening
        </span>

        <h1>
          Heart Risk Assessment
        </h1>

        <p>
          Enter your health and lifestyle
          information. The system will generate
          an ensemble prediction, SHAP
          explanation, visual SHAP plots, and a
          readable Explainable Generative AI
          report.
        </p>
      </section>

      <div className="assessment-layout">
        <form
          className="assessment-form"
          onSubmit={handleSubmit}
        >
          <section className="form-section">
            <div className="form-section-heading">
              <span>01</span>

              <div>
                <h2>
                  Basic information
                </h2>

                <p>
                  General information used by
                  the prediction model.
                </p>
              </div>
            </div>

            <div className="form-grid">
              <label>
                Age

                <input
                  type="number"
                  min="18"
                  max="55"
                  value={form.age}
                  onChange={(event) =>
                    updateNumber(
                      "age",
                      event.target.value
                    )
                  }
                  required
                />

                <small>
                  Accepted age range: 18–55
                  years
                </small>
              </label>

              <label>
                Sex

                <select
                  value={form.sex}
                  onChange={(event) =>
                    updateNumber(
                      "sex",
                      event.target.value
                    )
                  }
                >
                  <option value={0}>
                    Female
                  </option>

                  <option value={1}>
                    Male
                  </option>
                </select>
              </label>

              <label>
                BMI

                <input
                  type="number"
                  min="10"
                  max="70"
                  step="0.1"
                  value={form.bmi}
                  onChange={(event) =>
                    updateNumber(
                      "bmi",
                      event.target.value
                    )
                  }
                  required
                />

                <small>
                  Body Mass Index
                </small>
              </label>
            </div>
          </section>

          <section className="form-section">
            <div className="form-section-heading">
              <span>02</span>

              <div>
                <h2>
                  Medical information
                </h2>

                <p>
                  Select the health conditions
                  that apply.
                </p>
              </div>
            </div>

            <div className="form-grid">
              <label>
                High blood pressure

                <select
                  value={
                    form.high_blood_pressure
                  }
                  onChange={(event) =>
                    updateNumber(
                      "high_blood_pressure",
                      event.target.value
                    )
                  }
                >
                  <option value={0}>
                    No
                  </option>

                  <option value={1}>
                    Yes
                  </option>
                </select>
              </label>

              <label>
                High cholesterol

                <select
                  value={
                    form.high_cholesterol
                  }
                  onChange={(event) =>
                    updateNumber(
                      "high_cholesterol",
                      event.target.value
                    )
                  }
                >
                  <option value={0}>
                    No
                  </option>

                  <option value={1}>
                    Yes
                  </option>
                </select>
              </label>

              <label>
                Diabetes

                <select
                  value={form.diabetes}
                  onChange={(event) =>
                    updateNumber(
                      "diabetes",
                      event.target.value
                    )
                  }
                >
                  <option value={0}>
                    No
                  </option>

                  <option value={1}>
                    Yes
                  </option>
                </select>
              </label>

              <label>
                Time since cholesterol check

                <select
                  value={
                    form
                      .time_since_cholesterol_check
                  }
                  onChange={(event) =>
                    updateNumber(
                      "time_since_cholesterol_check",
                      event.target.value
                    )
                  }
                >
                  <option value={1}>
                    Within the past year
                  </option>

                  <option value={2}>
                    Within the past two years
                  </option>

                  <option value={3}>
                    Within the past five years
                  </option>

                  <option value={4}>
                    More than five years ago
                  </option>

                  <option value={5}>
                    Never checked
                  </option>
                </select>
              </label>
            </div>
          </section>

          <section className="form-section">
            <div className="form-section-heading">
              <span>03</span>

              <div>
                <h2>
                  Lifestyle information
                </h2>

                <p>
                  Smoking, exercise, and mental
                  wellbeing factors.
                </p>
              </div>
            </div>

            <div className="form-grid">
              <label>
                Smoking history

                <select
                  value={
                    form.smoking_history
                  }
                  onChange={(event) =>
                    updateNumber(
                      "smoking_history",
                      event.target.value
                    )
                  }
                >
                  <option value={0}>
                    No
                  </option>

                  <option value={1}>
                    Yes
                  </option>
                </select>
              </label>

              <label>
                Current smoking

                <select
                  value={
                    form.current_smoking
                  }
                  onChange={(event) =>
                    updateNumber(
                      "current_smoking",
                      event.target.value
                    )
                  }
                >
                  <option value={0}>
                    No
                  </option>

                  <option value={1}>
                    Yes
                  </option>

                  <option value={2}>
                    Occasional or other
                    category
                  </option>
                </select>
              </label>

              <label>
                Regular exercise

                <select
                  value={form.exercise}
                  onChange={(event) =>
                    updateNumber(
                      "exercise",
                      event.target.value
                    )
                  }
                >
                  <option value={1}>
                    Yes
                  </option>

                  <option value={0}>
                    No
                  </option>
                </select>
              </label>

              <label>
                Mentally unhealthy days

                <input
                  type="number"
                  min="0"
                  max="30"
                  value={
                    form
                      .mentally_unhealthy_days
                  }
                  onChange={(event) =>
                    updateNumber(
                      "mentally_unhealthy_days",
                      event.target.value
                    )
                  }
                  required
                />

                <small>
                  Number of days during the
                  last 30 days
                </small>
              </label>
            </div>
          </section>

          <section className="form-section">
            <div className="form-section-heading">
              <span>04</span>

              <div>
                <h2>
                  Dietary survey information
                </h2>

                <p>
                  These values follow the
                  coding expected by the
                  trained survey-based model.
                </p>
              </div>
            </div>

            <div className="form-grid">
              <label>
                Alcohol consumption value

                <input
                  type="number"
                  min="0"
                  value={
                    form
                      .alcohol_consumption
                  }
                  onChange={(event) =>
                    updateNumber(
                      "alcohol_consumption",
                      event.target.value
                    )
                  }
                  required
                />
              </label>

              <label>
                Fruit intake value

                <input
                  type="number"
                  min="0"
                  value={
                    form.fruit_intake
                  }
                  onChange={(event) =>
                    updateNumber(
                      "fruit_intake",
                      event.target.value
                    )
                  }
                  required
                />
              </label>

              <label>
                Fruit juice intake value

                <input
                  type="number"
                  min="0"
                  value={
                    form
                      .fruit_juice_intake
                  }
                  onChange={(event) =>
                    updateNumber(
                      "fruit_juice_intake",
                      event.target.value
                    )
                  }
                  required
                />
              </label>

              <label>
                Other vegetable intake value

                <input
                  type="number"
                  min="0"
                  value={
                    form
                      .other_vegetable_intake
                  }
                  onChange={(event) =>
                    updateNumber(
                      "other_vegetable_intake",
                      event.target.value
                    )
                  }
                  required
                />
              </label>

              <label>
                Green vegetable intake value

                <input
                  type="number"
                  min="0"
                  value={
                    form
                      .green_vegetable_intake
                  }
                  onChange={(event) =>
                    updateNumber(
                      "green_vegetable_intake",
                      event.target.value
                    )
                  }
                  required
                />
              </label>
            </div>
          </section>

          {error && (
            <div className="alert alert-error">
              {error}
            </div>
          )}

          <button
            className="prediction-submit-button"
            type="submit"
            disabled={submitting}
          >
            {submitting ? (
              "Generating prediction and SHAP plots..."
            ) : (
              <>
                Generate prediction
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </form>

        <aside className="assessment-information">
          <BrainCircuit size={30} />

          <h2>
            How your result is generated
          </h2>

          <ol>
            <li>
              Submitted information is
              converted into the model’s
              required features.
            </li>

            <li>
              Logistic Regression, Random
              Forest, and XGBoost calculate
              separate probabilities.
            </li>

            <li>
              The weighted ensemble creates
              the final predicted probability.
            </li>

            <li>
              SHAP identifies factors that
              influenced the XGBoost
              component.
            </li>

            <li>
              Waterfall and contribution plots
              visually display the SHAP
              evidence.
            </li>

            <li>
              Gemini converts the structured
              SHAP evidence into readable
              language.
            </li>
          </ol>

          <div className="assessment-note">
            SHAP explains model behaviour. It
            does not prove that a feature
            medically caused cardiovascular
            disease.
          </div>
        </aside>
      </div>

      {result && (
        <section
          id="prediction-result"
          className="prediction-result"
        >
          <div className="result-heading">
            <div>
              <span className="eyebrow">
                Assessment completed
              </span>

              <h2>
                Your prediction result
              </h2>
            </div>

            <span className="result-status">
              <CheckCircle2 size={18} />
              Saved securely
            </span>
          </div>

          <div className="risk-summary-card">
            <div
              className="probability-circle"
              style={{
                background: `conic-gradient(
                  #0b8f78 ${
                    result
                      .probability_percent *
                    3.6
                  }deg,
                  #e7eeec 0deg
                )`,
              }}
            >
              <div>
                <strong>
                  {
                    result
                      .probability_percent
                  }
                  %
                </strong>

                <span>
                  Predicted probability
                </span>
              </div>
            </div>

            <div className="risk-summary-content">
              <span className="risk-label">
                {result.risk_category}
              </span>

              <h3>
                Ensemble risk assessment
              </h3>

              <p>
                This probability was calculated
                by combining Logistic
                Regression, Random Forest, and
                XGBoost.
              </p>

              <div className="threshold-list">
                <span>
                  Screening threshold:{" "}
                  {(
                    result.thresholds
                      .screening * 100
                  ).toFixed(1)}
                  %
                </span>

                <span>
                  High-specificity threshold:{" "}
                  {(
                    result.thresholds
                      .high_specificity * 100
                  ).toFixed(1)}
                  %
                </span>
              </div>
            </div>
          </div>

          <div className="factor-grid">
            <article className="factor-card increasing-card">
              <div className="factor-heading">
                <TrendingUp size={23} />

                <div>
                  <h3>
                    Factors increasing the
                    prediction
                  </h3>

                  <p>
                    Positive SHAP contributions
                    for this result
                  </p>
                </div>
              </div>

              {result.xai_explanation
                .increasing_factors.length >
              0 ? (
                <ul className="factor-list">
                  {result.xai_explanation
                    .increasing_factors
                    .map((factor) => (
                      <li
                        key={
                          factor.feature
                        }
                      >
                        <span>
                          {readableFeatureName(
                            factor.feature
                          )}
                        </span>

                        <strong>
                          +
                          {factor.shap_value.toFixed(
                            3
                          )}
                        </strong>
                      </li>
                    ))}
                </ul>
              ) : (
                <p>
                  No major increasing factors
                  were identified.
                </p>
              )}
            </article>

            <article className="factor-card decreasing-card">
              <div className="factor-heading">
                <TrendingDown size={23} />

                <div>
                  <h3>
                    Factors decreasing the
                    prediction
                  </h3>

                  <p>
                    Negative SHAP contributions
                    for this result
                  </p>
                </div>
              </div>

              {result.xai_explanation
                .decreasing_factors.length >
              0 ? (
                <ul className="factor-list">
                  {result.xai_explanation
                    .decreasing_factors
                    .map((factor) => (
                      <li
                        key={
                          factor.feature
                        }
                      >
                        <span>
                          {readableFeatureName(
                            factor.feature
                          )}
                        </span>

                        <strong>
                          {factor.shap_value.toFixed(
                            3
                          )}
                        </strong>
                      </li>
                    ))}
                </ul>
              ) : (
                <p>
                  No major decreasing factors
                  were identified.
                </p>
              )}
            </article>
          </div>

          <section className="shap-visualization-section">
            <div className="report-section-heading">
              <div>
                <span className="eyebrow">
                  Visual model explanation
                </span>

                <h2>
                  SHAP visualizations
                </h2>
              </div>

              <BrainCircuit size={27} />
            </div>

            <p className="shap-visualization-note">
              These plots explain how
              individual features influenced
              the XGBoost component of the
              ensemble prediction.
            </p>

            <div className="shap-image-grid">
              <article className="shap-image-card">
                <div>
                  <h3>
                    Waterfall plot
                  </h3>

                  <p>
                    Shows how feature
                    contributions moved the
                    XGBoost output away from
                    its baseline value.
                  </p>
                </div>

                <img
                  src={
                    result
                      .shap_visualizations
                      .waterfall_image
                  }
                  alt="SHAP waterfall plot for the XGBoost prediction"
                />
              </article>

              <article className="shap-image-card">
                <div>
                  <h3>
                    Feature contribution plot
                  </h3>

                  <p>
                    Compares the strongest
                    positive and negative SHAP
                    contributions.
                  </p>
                </div>

                <img
                  src={
                    result
                      .shap_visualizations
                      .bar_image
                  }
                  alt="SHAP feature contribution bar plot"
                />
              </article>
            </div>

            <div className="assessment-note shap-warning-note">
              {
                result
                  .shap_visualizations
                  .important_note
              }
            </div>
          </section>

          <article className="xgenai-report-card">
            <div className="factor-heading">
              <BrainCircuit size={26} />

              <div>
                <span className="eyebrow">
                  Explainable Generative AI
                </span>

                <h3>
                  Personalized explanation
                </h3>
              </div>
            </div>

            <div className="markdown-report">
              <ReactMarkdown>
                {
                  result
                    .xgenai_explanation
                }
              </ReactMarkdown>
            </div>
          </article>

          <div className="medical-disclaimer">
            <ShieldCheck size={21} />

            <p>
              This result is intended for
              research and preliminary
              screening only. It is not a
              medical diagnosis and should not
              replace professional medical
              advice.
            </p>
          </div>
        </section>
      )}
    </AppLayout>
  );
}