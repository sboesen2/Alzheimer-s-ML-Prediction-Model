import React from 'react';
import { Typography, Card, CardContent } from '@mui/material';

const Insights = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>Insights on Alzheimer's Risk Factors</Typography>
        <Typography paragraph>
          Genetic factors play a significant role in Alzheimer's risk. SNPs (Single Nucleotide Polymorphisms)
          are variations in single DNA building blocks that can influence disease risk.
        </Typography>
        <Typography paragraph>
          The SNP on chromosome 9 (chr9:92808229-A) has been associated with...
          {/* Add more educational content */}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default Insights;