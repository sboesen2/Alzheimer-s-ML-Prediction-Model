import React from 'react';
import { Typography, Button, Box, Container, Grid, Card, CardContent, Paper } from '@mui/material';
import { Link } from 'react-router-dom';
import { styled } from '@mui/system';
import ComputerIcon from '@mui/icons-material/Computer';
import SearchIcon from '@mui/icons-material/Search';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

const HeroSection = styled(Box)(({ theme }) => ({
  backgroundImage: 'url("/hero-image.jpg")',
  backgroundSize: 'cover',
  backgroundPosition: 'center',
  height: '65vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: 'white',
  textAlign: 'center',
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)', // Dark overlay
  },
}));

const FeatureCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-10px)',
  },
}));

const features = [
  { title: 'Cutting-edge AI', icon: <ComputerIcon fontSize="large" />, description: 'Utilize advanced machine learning algorithms for accurate predictions.' },
  { title: 'Genetic Markers', icon: <SearchIcon fontSize="large" />, description: 'Analyze specific genetic markers associated with Alzheimer\'s risk.' },
  { title: 'Early Detection', icon: <AccessTimeIcon fontSize="large" />, description: 'Identify potential risks early for better prevention and management.' },
];

function Home() {
  return (
    <Box>
      <HeroSection>
        <Container sx={{ position: 'relative', zIndex: 1 }}>
          <Typography variant="h2" component="h1" gutterBottom>
            Predict Alzheimer's Risk with AI
          </Typography>
          <Typography variant="h5" paragraph>
            Advanced genetic analysis for early detection
          </Typography>
          <Button
            variant="contained"
            color="secondary"
            component={Link}
            to="/predict"
            size="large"
            sx={{ mt: 2 }}
          >
            Start Your Analysis
          </Button>
        </Container>
      </HeroSection>

      <Container sx={{ mt: 8, mb: 8 }}>
        <Grid container spacing={4}>
          {features.map((feature) => (
            <Grid item xs={12} md={4} key={feature.title}>
              <FeatureCard>
                <CardContent>
                  {feature.icon}
                  <Typography variant="h5" component="div" gutterBottom sx={{ mt: 2 }}>
                    {feature.title}
                  </Typography>
                  <Typography variant="body2">
                    {feature.description}
                  </Typography>
                </CardContent>
              </FeatureCard>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* New sections start here */}
      <Container sx={{ mt: 8, mb: 8 }}>
        <Typography variant="h4" gutterBottom>About the Project</Typography>
        <Typography paragraph>
          This Alzheimer's Risk Prediction tool uses advanced machine learning algorithms to analyze genetic markers and predict the risk of Alzheimer's disease. My goal is to provide early detection capabilities to improve prevention and management strategies.
        </Typography>

        <Typography variant="h4" gutterBottom sx={{ mt: 4 }}>How It Works</Typography>
        <Paper elevation={3} sx={{ p: 3 }}>
          <ol>
            <li>Input your genetic markers</li>
            <li>The AI analyzes the data</li>
            <li>Receive a personalized risk assessment</li>
            <li>Consult with healthcare professionals for next steps</li>
          </ol>
        </Paper>

        <Typography variant="h4" gutterBottom sx={{ mt: 4 }}>Core Belief</Typography>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="body1" paragraph>
            "The real voyage of discovery consists not in seeking new landscapes, but in having new eyes."
          </Typography>
          <Typography variant="subtitle2">- Marcel Proust, Novelist</Typography>
        </Paper>
      </Container>
    </Box>
  );
}

export default Home;