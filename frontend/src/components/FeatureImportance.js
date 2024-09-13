import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Dialog, DialogTitle, DialogContent, Typography, Button, Box, useTheme } from '@mui/material';

const FeatureImportance = () => {
  const [featureImportance, setFeatureImportance] = useState([]);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const theme = useTheme();

  useEffect(() => {
    const fetchFeatureImportance = async () => {
      try {
        const response = await axios.get('https://alzheimers-backend-925506798704.us-central1.run.app/feature_importance');
        setFeatureImportance(response.data.sort((a, b) => b.importance - a.importance));
      } catch (error) {
        console.error('Error fetching feature importance:', error);
      }
    };

    fetchFeatureImportance();
  }, []);

  const truncateFeatureName = (name, maxLength = 20) => {
    return name.length > maxLength ? name.substring(0, maxLength) + '...' : name;
  };

  const handleBarClick = (data) => {
    setSelectedFeature(data);
  };

  const handleCloseDialog = () => {
    setSelectedFeature(null);
  };

  const getFeatureDescription = (feature) => {
    const descriptions = {
      'STRONGEST SNP-RISK ALLELE': 'This feature represents the most significant Single Nucleotide Polymorphism (SNP) associated with Alzheimer\'s risk.',
      'P-VALUE': 'The p-value indicates the statistical significance of the association between the genetic variant and Alzheimer\'s disease.',
      'OR or BETA': 'This represents the effect size of the genetic variant, indicating the strength of its association with Alzheimer\'s risk.',
      'RISK ALLELE FREQUENCY': 'This is the frequency of the risk-associated allele in the population studied.',
      'PVALUE_MLOG': 'This is the negative log10 of the p-value, which helps visualize very small p-values more effectively.'
    };
    return descriptions[feature] || 'No detailed description available for this feature.';
  };

  return (
    <Box sx={{ width: '100%', height: 600, mt: 4, mb: 4 }}>
      <Typography variant="h5" gutterBottom sx={{ color: theme.palette.primary.main }}>
        Feature Importance
      </Typography>
      <ResponsiveContainer>
        <BarChart
          layout="vertical"
          data={featureImportance}
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <XAxis
            type="number"
            tick={{ fill: theme.palette.text.primary }}
            axisLine={{ stroke: theme.palette.divider }}
          />
          <YAxis
            dataKey="feature"
            type="category"
            width={140}
            tick={{ fontSize: 12, fill: theme.palette.text.primary }}
            tickFormatter={truncateFeatureName}
            axisLine={{ stroke: theme.palette.divider }}
          />
          <Tooltip
            formatter={(value, name, props) => [value.toFixed(4), props.payload.feature]}
            labelFormatter={(label) => `Feature: ${label}`}
            contentStyle={{
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: '4px',
              color: theme.palette.text.primary
            }}
          />
          <Bar
            dataKey="importance"
            fill={theme.palette.primary.main}
            onClick={handleBarClick}
            cursor="pointer"
          />
        </BarChart>
      </ResponsiveContainer>

      <Dialog open={!!selectedFeature} onClose={handleCloseDialog}>
        <DialogTitle sx={{ backgroundColor: theme.palette.background.paper, color: theme.palette.text.primary }}>
          {selectedFeature?.feature}
        </DialogTitle>
        <DialogContent sx={{ backgroundColor: theme.palette.background.paper, color: theme.palette.text.primary }}>
          <Typography variant="body1" paragraph>
            Importance: {selectedFeature?.importance.toFixed(4)}
          </Typography>
          <Typography variant="body1" paragraph>
            {getFeatureDescription(selectedFeature?.feature)}
          </Typography>
          <Button variant="contained" onClick={handleCloseDialog} sx={{ mt: 2 }}>Close</Button>
        </DialogContent>
      </Dialog>

      <Typography variant="body2" sx={{ mt: 2, color: theme.palette.text.secondary }}>
        {/*Feature importance shows which inputs have the most significant overall impact on the model's predictions across all data.
        This helps us understand which genetic factors are generally most influential in determining Alzheimer's risk.

        -This is the piece of code that is the reason why those two text boxes kept overlapping. there was one on the
        predict.js on one on here so this one had to be removed so the whole thing wouldnt get deleted*/}
      </Typography>
    </Box>
  );
};

export default FeatureImportance;