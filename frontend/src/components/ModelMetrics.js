import React from 'react';
import { Typography, Grid, Paper } from '@mui/material';

const ModelMetrics = ({ accuracy, sensitivity, specificity }) => {
  return (
    <Paper elevation={3} style={{ padding: '20px' }}>
      <Typography variant="h6" gutterBottom>Model Performance Metrics</Typography>
      <Grid container spacing={2}>
        <Grid item xs={4}>
          <Typography>Accuracy: {accuracy}%</Typography>
        </Grid>
        <Grid item xs={4}>
          <Typography>Sensitivity: {sensitivity}%</Typography>
        </Grid>
        <Grid item xs={4}>
          <Typography>Specificity: {specificity}%</Typography>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default ModelMetrics;