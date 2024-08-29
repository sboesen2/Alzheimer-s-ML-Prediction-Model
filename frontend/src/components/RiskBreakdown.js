import React, { useState } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Legend } from 'recharts';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography, Button, Box } from '@mui/material';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d', '#ffc658'];

const RiskBreakdown = ({ features, totalRisk }) => {
  const [showAllFeatures, setShowAllFeatures] = useState(false);

  // Filter out zero contributions and sort by relative importance
  const significantFeatures = features
    .filter(f => f.relativeImportance > 0)
    .sort((a, b) => b.relativeImportance - a.relativeImportance);

  // Take top 4 and group the rest as "Other"
  const topFeatures = significantFeatures.slice(0, 4);
  const otherFeatures = significantFeatures.slice(4);

  const chartData = [
    ...topFeatures,
    {
      name: 'Other',
      relativeImportance: otherFeatures.reduce((sum, f) => sum + f.relativeImportance, 0),
      contribution: otherFeatures.reduce((sum, f) => sum + f.contribution, 0),
    }
  ];

  const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <Paper
        elevation={3}
        style={{
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '10px',
          border: '1px solid #ccc',
          zIndex: 1000
        }}
      >
        <Typography variant="body2" style={{ color: 'white' }}>{`${data.name}`}</Typography>
        <Typography variant="body2" style={{ color: 'white' }}>{`Relative Importance: ${(data.relativeImportance * 100).toFixed(2)}%`}</Typography>
        {data.contribution !== undefined &&
          <Typography variant="body2" style={{ color: 'white' }}>{`Contribution: ${(data.contribution * 100).toFixed(2)}%`}</Typography>
        }
      </Paper>
    );
  }
  return null;
};

  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index }) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <Paper elevation={3} style={{ padding: '20px', marginTop: '20px' }}>
      <Typography variant="h6" gutterBottom>Risk Factor Breakdown</Typography>
      <Typography variant="body1" gutterBottom>Total Risk: {(totalRisk * 100).toFixed(2)}%</Typography>

      <Box display="flex" justifyContent="space-between" flexWrap="wrap">
        <Box width="48%" height={450} mb={2}>
          <ResponsiveContainer>
            <PieChart>
              <Pie
                data={chartData}
                dataKey="relativeImportance"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={150}
                fill="#8884d8"
                labelLine={false}
                label={renderCustomizedLabel}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} wrapperStyle={{ zIndex: 1000 }} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Box>

        <Box width="50%" height={450} mb={2}>
          <ResponsiveContainer>
            <BarChart
              data={showAllFeatures ? significantFeatures : topFeatures}
              layout="vertical"
              margin={{ top: 5, right: 1, left: 40, bottom: 5 }}
            >
              <XAxis type="number" />
              <YAxis dataKey="name" type="category" width={180} />
              <Tooltip content={<CustomTooltip />} wrapperStyle={{ zIndex: 1000 }} />
              <Legend />
              <Bar dataKey="relativeImportance" fill="#8884d8" name="Relative Importance" cursor="pointer" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </Box>

      <Button
        variant="contained"
        color="primary"
        onClick={() => setShowAllFeatures(!showAllFeatures)}
        style={{ marginTop: '20px' }}
      >
        {showAllFeatures ? 'Show Top Features' : 'Show All Features'}
      </Button>

      <TableContainer component={Paper} style={{ marginTop: '20px' }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Feature</TableCell>
              <TableCell align="right">Contribution</TableCell>
              <TableCell align="right">Relative Importance</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(showAllFeatures ? significantFeatures : topFeatures).map((feature, index) => (
              <TableRow key={index}>
                <TableCell component="th" scope="row">
                  {feature.name}
                </TableCell>
                <TableCell align="right">{(feature.contribution * 100).toFixed(2)}%</TableCell>
                <TableCell align="right">{(feature.relativeImportance * 100).toFixed(2)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default RiskBreakdown;