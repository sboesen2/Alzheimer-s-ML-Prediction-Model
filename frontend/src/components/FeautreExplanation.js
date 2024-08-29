import React from 'react';
import { Typography, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const featureExplanations = {
  'STRONGEST SNP-RISK ALLELE_chr9:92808229-A': 'This SNP is located on chromosome 9 and is associated with...',
  'P-VALUE_M': 'Represents the statistical significance of the genetic association...',
  // Add more explanations for other features
};

const FeatureExplanation = () => {
  return (
    <div>
      <Typography variant="h6" gutterBottom>Feature Explanations</Typography>
      {Object.entries(featureExplanations).map(([feature, explanation]) => (
        <Accordion key={feature}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>{feature}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography>{explanation}</Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </div>
  );
};

export default FeatureExplanation;