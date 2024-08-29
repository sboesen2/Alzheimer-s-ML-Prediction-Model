import React from 'react';
import InfoSection from './InfoSection';

const PredictionTool = () => {
  return (
    <div>
      {/* Input fields */}
      <div>
        <label>STRONGEST SNP-RISK ALLELE</label>
        <input type="text" name="snpRiskAllele" />
        <InfoSection
          title="What is STRONGEST SNP-RISK ALLELE?"
          content="The strongest SNP-risk allele refers to the variant of the genetic marker that is most strongly associated with an increased risk for Alzheimer's disease."
        />
      </div>

      <div>
        <label>P-VALUE</label>
        <input type="text" name="pValue" />
        <InfoSection
          title="What is P-VALUE?"
          content="The p-value indicates the strength of the evidence against the null hypothesis. A lower p-value suggests that the association between the SNP and the disease is statistically significant."
        />
      </div>

      <div>
        <label>OR or BETA</label>
        <input type="text" name="orBeta" />
        <InfoSection
          title="What is OR or BETA?"
          content="The OR (Odds Ratio) or BETA value represents the effect size, indicating how strongly the SNP is associated with the disease."
        />
      </div>

      {/* Prediction output */}
      <div>
        <h2>Predicted Risk: 6.08%</h2>
        <InfoSection
          title="Understanding the Predicted Risk"
          content="This percentage represents the likelihood of developing Alzheimer's disease based on the genetic data provided."
        />
      </div>

      {/* SHAP Values and Feature Importance */}
      <div>
        <h2>Top 15 SHAP Values</h2>
        {/* SHAP values chart */}
        <InfoSection
          title="What are SHAP Values?"
          content="SHAP values indicate the contribution of each feature to the prediction, helping to explain how the model arrived at the final risk score."
        />
      </div>

      <div>
        <h2>Feature Importance</h2>
        {/* Feature importance chart */}
        <InfoSection
          title="Understanding Feature Importance"
          content="Feature importance shows which features have the most influence on the model's predictions. Higher values mean greater influence."
        />
      </div>
    </div>
  );
};

export default PredictionTool;
