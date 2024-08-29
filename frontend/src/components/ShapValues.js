import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useTheme } from '@mui/material/styles';
import { Typography, Box, Popover, CircularProgress } from '@mui/material';

const ShapValues = ({ shapValues, featureNames }) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = useState(null);
  const [popoverContent, setPopoverContent] = useState(null);
  const [data, setData] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    console.log('ShapValues received:', { shapValues, featureNames });
    setLoading(true);
    setError(null);

    if (!Array.isArray(shapValues) || !Array.isArray(featureNames) || shapValues.length !== featureNames.length) {
      console.error('Invalid shapValues or featureNames', { shapValues, featureNames });
      setError('Invalid data received for SHAP values');
      setLoading(false);
      return;
    }

    try {
      const processedData = shapValues
        .map((value, index) => ({
          feature: featureNames[index],
          value: parseFloat(value)
        }))
        .filter(item => !isNaN(item.value))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 15);

      console.log('Processed SHAP data:', processedData);
      setData(processedData);
    } catch (err) {
      console.error('Error processing SHAP values', err);
      setError('Error processing SHAP values');
    }

    setLoading(false);
  }, [shapValues, featureNames]);

  if (loading) {
    return <CircularProgress />;
  }

  if (error) {
    return <Typography color="error">{error}</Typography>;
  }

  if (data.length === 0) {
    return <Typography>No SHAP values to display</Typography>;
  }

  const maxValue = Math.max(...data.map(d => Math.abs(d.value)));

  const handleBarClick = (entry, index, event) => {
    setAnchorEl(event.currentTarget);
    setPopoverContent(
      <Box>
        <Typography variant="subtitle1">{entry.feature}</Typography>
        <Typography variant="body2">SHAP Value: {entry.value.toFixed(4)}</Typography>
        <Typography variant="body2">
          This feature {entry.value >= 0 ? 'increases' : 'decreases'} the risk prediction by {Math.abs(entry.value).toFixed(4)}.
        </Typography>
      </Box>
    );
  };

  const handlePopoverClose = () => {
    setAnchorEl(null);
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <Box sx={{ bgcolor: 'background.paper', p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
          <Typography variant="subtitle2">{label}</Typography>
          <Typography variant="body2">SHAP Value: {payload[0].value.toFixed(4)}</Typography>
        </Box>
      );
    }
    return null;
  };

  return (
    <Box sx={{ width: '100%', height: 600, mt: 4, mb: 4 }}>
      <Typography variant="h5" gutterBottom sx={{ color: theme.palette.primary.main }}>
        Top 15 SHAP Values
      </Typography>
      <ResponsiveContainer>
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 5, right: 30, left: 250, bottom: 5 }}
        >
          <XAxis
            type="number"
            domain={[-maxValue, maxValue]}
            tick={{ fill: theme.palette.text.primary }}
            axisLine={{ stroke: theme.palette.divider }}
          />
          <YAxis
            dataKey="feature"
            type="category"
            width={240}
            tick={{ fontSize: 12, fill: theme.palette.text.primary }}
            axisLine={{ stroke: theme.palette.divider }}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine x={0} stroke={theme.palette.divider} />
          <Bar
            dataKey="value"
            fill={(entry) => (entry.value >= 0 ? theme.palette.primary.main : theme.palette.secondary.main)}
            onClick={handleBarClick}
            cursor="pointer"
          />
        </BarChart>
      </ResponsiveContainer>
      <Typography variant="body2" sx={{ mt: 2, color: theme.palette.text.secondary }}>
      </Typography>
      <Popover
        open={Boolean(anchorEl)}
        anchorEl={anchorEl}
        onClose={handlePopoverClose}
        anchorOrigin={{
          vertical: 'center',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'center',
          horizontal: 'left',
        }}
      >
        {popoverContent}
      </Popover>
    </Box>
  );
};

export default ShapValues;