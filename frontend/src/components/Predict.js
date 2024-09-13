import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { TextField, Button, Typography, Box, Paper, CircularProgress, Slider, Grid, Tooltip, IconButton, Snackbar } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import FeatureImportance from './FeatureImportance';
import ShapValues from './ShapValues';
import RiskBreakdown from './RiskBreakdown';
import { styled, useTheme } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginTop: theme.spacing(3),
  backgroundColor: theme.palette.background.paper,
  color: theme.palette.text.primary,
}));

const InfoTooltip = ({ title }) => (
  <Tooltip title={title}>
    <IconButton size="small">
      <InfoIcon fontSize="small" color="primary" />
    </IconButton>
  </Tooltip>
);

function Predict() {
  const theme = useTheme();
  const [researcherInputs, setResearcherInputs] = useState({
    snpRiskAllele: '',
    pValue: '',
    orBeta: '',
    riskAlleleFrequency: '',
    pValueMlog: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [shapValues, setShapValues] = useState(null);
  const [featureNames, setFeatureNames] = useState(null);
  const [riskBreakdown, setRiskBreakdown] = useState(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  useEffect(() => {
    fetchSampleData();
  }, []);

  const fetchSampleData = async () => {
    try {
      const response = await axios.get('https://alzheimers-backend-925506798704.us-central1.run.app/predict');
      setResearcherInputs(response.data);
      setSnackbarMessage('Sample data loaded successfully');
      setSnackbarOpen(true);
    } catch (err) {
      console.error('Error fetching sample data:', err);
      setSnackbarMessage('Failed to load sample data');
      setSnackbarOpen(true);
    }
  };

  const handleResearcherInputChange = (e) => {
    setResearcherInputs({ ...researcherInputs, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('https://alzheimers-backend-925506798704.us-central1.run.app/predict', researcherInputs);
      setPrediction(response.data.risk);
      setShapValues(response.data.shap_values);
      setFeatureNames(response.data.feature_names);
      setRiskBreakdown(response.data.riskBreakdown);
      setSnackbarMessage('Prediction completed successfully');
      setSnackbarOpen(true);
    } catch (err) {
      setError(`An error occurred: ${err.response ? err.response.data.error : err.message}`);
      setSnackbarMessage('Error occurred during prediction');
      setSnackbarOpen(true);
    }
    setLoading(false);
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  return (
    <StyledPaper elevation={3}>
      <Typography variant="h4" gutterBottom color="primary">Alzheimer's Risk Prediction Tool</Typography>

      <Box mt={3} component="form" onSubmit={handleSubmit}>
        <Typography variant="body1" gutterBottom>
          Input the following genetic data features used in our model:
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(researcherInputs).map(([key, value]) => (
            <Grid item xs={12} sm={6} key={key}>
              <TextField
                fullWidth
                label={key.toUpperCase().replace('_', ' ')}
                name={key}
                value={value}
                onChange={handleResearcherInputChange}
                margin="normal"
                InputLabelProps={{
                  style: { color: theme.palette.text.secondary }
                }}
                InputProps={{
                  style: { color: theme.palette.text.primary },
                  endAdornment: <InfoTooltip title={getTooltipText(key)} />
                }}
              />
            </Grid>
          ))}
        </Grid>
      </Box>

      <Box mt={2} display="flex" justifyContent="space-between">
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Predict Risk'}
        </Button>
        <Button
          variant="outlined"
          color="secondary"
          onClick={fetchSampleData}
        >
          Load Sample Data
        </Button>
      </Box>

      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}

      {prediction !== null && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="h6">
            Predicted Risk: {prediction.toFixed(2)}%
          </Typography>
          <Box sx={{ width: '100%', mt: 1 }}>
            <Slider
              value={prediction}
              min={0}
              max={100}
              disabled
              valueLabelDisplay="auto"
              sx={{
                '& .MuiSlider-thumb': {
                  color: prediction > 50 ? theme.palette.error.main : theme.palette.success.main,
                },
                '& .MuiSlider-track': {
                  color: prediction > 50 ? theme.palette.error.main : theme.palette.success.main,
                },
              }}
            />
          </Box>
          <Typography variant="body2" sx={{ mt: 1 }}>
            {prediction <= 30 ? 'Low risk: Maintain a healthy lifestyle to keep your risk low.' :
             prediction <= 60 ? 'Moderate risk: Consider lifestyle changes and regular check-ups.' :
             'High risk: Consult with a healthcare professional for personalized advice.'}
          </Typography>
          {riskBreakdown && (
            <RiskBreakdown features={riskBreakdown.features} totalRisk={riskBreakdown.totalRisk} />
          )}
          {shapValues && featureNames && (
            <>
              <Typography variant="h6" sx={{ mt: 3 }}>SHAP Values Explanation</Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                SHAP (SHapley Additive exPlanations) values help us understand how each feature contributes to the prediction for this specific input. Positive values increase the risk prediction, while negative values decrease it.
              </Typography>
              <ShapValues shapValues={shapValues} featureNames={featureNames} />
              <Typography variant="body2" sx={{ mt: 1 }}>
                The chart above shows how each feature impacts the models output. Bars extending to the right increase the risk prediction, while bars to the left decrease it. The length of each bar indicates the magnitude of the features impact.
              </Typography>
            </>
          )}
        </Box>
      )}

      <Box sx={{ mt: 4 }}>
        <Typography variant="h6">Feature Importance</Typography>
        <Typography variant="body2" sx={{ mt: 1, mb: 2 }}>
          Feature importance shows which inputs have the most significant overall impact on the model's predictions across all data. This helps us understand which genetic factors are generally most influential in determining Alzheimer's risk.
        </Typography>
        <FeatureImportance />
      </Box>

      <Typography variant="body2" sx={{ mt: 2 }}>
        Note: This tool is for research and educational purposes only. It uses a machine learning model trained on genetic data to estimate Alzheimer's risk based on specific genetic markers. The prediction is based on population-level statistics and may not accurately reflect an individual's actual risk. For medical advice, please consult a healthcare professional.
      </Typography>

      <Snackbar
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={snackbarMessage}
        ContentProps={{
          sx: { backgroundColor: theme.palette.background.paper, color: theme.palette.text.primary }
        }}
      />
    </StyledPaper>
  );
}

function getTooltipText(key) {
  const tooltips = {
    snpRiskAllele: "The strongest Single Nucleotide Polymorphism (SNP) associated with Alzheimer's risk. It represents the genetic variant most strongly linked to the disease.",
    pValue: "The statistical significance of the genetic association. Lower p-values indicate stronger evidence against the null hypothesis of no association.",
    orBeta: "Odds Ratio (OR) or Beta coefficient, indicating the effect size of the genetic variant. An OR > 1 suggests increased risk, while OR < 1 suggests decreased risk.",
    riskAlleleFrequency: "The frequency of the risk-associated allele in the population. Higher frequencies may indicate a more common genetic risk factor.",
    pValueMlog: "The negative log (base 10) of the p-value. This transformation enhances the scale for very small p-values, making them easier to compare and visualize."
  };
  return tooltips[key] || "No description available";
}

export default Predict;