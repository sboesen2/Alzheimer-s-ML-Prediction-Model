import React, { useState } from 'react';
import { Typography, Container, Box, Grid, Avatar, Button, Divider, Card, CardContent, Link, Tooltip, Zoom, Fade, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { GitHub, LinkedIn, Email, KeyboardArrowDown, CheckCircleOutline } from '@mui/icons-material';
import { styled } from '@mui/system';

const StyledAvatar = styled(Avatar)(({ theme }) => ({
  width: theme.spacing(25),
  height: theme.spacing(25),
  margin: 'auto',
  border: `4px solid ${theme.palette.primary.main}`,
  boxShadow: theme.shadows[10],
  transition: 'transform 0.3s ease-in-out',
  '&:hover': {
    transform: 'scale(1.05)',
  },
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  marginTop: theme.spacing(6),
  marginBottom: theme.spacing(3),
  fontWeight: 'bold',
  color: theme.palette.primary.main,
  textAlign: 'center',
}));

const FeatureCard = React.memo(styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-10px)',
    boxShadow: theme.shadows[20],
  },
})));

const SocialButton = styled(Button)(({ theme }) => ({
  margin: theme.spacing(0, 1),
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'scale(1.05)',
  },
}));

function Author() {
  const [showContent, setShowContent] = useState(false);

  const scrollToContent = () => {
    setShowContent(true);
    const contentElement = document.getElementById('author-content');
    if (contentElement) {
      contentElement.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <Container maxWidth="md">
      <Box my={6} display="flex" flexDirection="column" alignItems="center">
        <Fade in={true} timeout={1000}>
          <Typography variant="h2" align="center" gutterBottom color="primary">
            About the Author and Project
          </Typography>
        </Fade>

        <Fade in={true} timeout={1500}>
          <StyledAvatar alt="Sam Boesen" src="IMG_4540.jpg" />
        </Fade>
        <Fade in={true} timeout={2000}>
          <Typography variant="h3" align="center" gutterBottom sx={{ mt: 3 }}>
            Sam Boesen
          </Typography>
        </Fade>
        <Fade in={true} timeout={2500}>
          <Typography variant="h6" align="center" gutterBottom color="textSecondary">
            BioPharma | Machine Learning | Alzheimer's Research
          </Typography>
        </Fade>
        <Fade in={true} timeout={3000}>
          <Box display="flex" justifyContent="center" mt={3}>
            <Tooltip title="View GitHub Profile" arrow TransitionComponent={Zoom}>
              <SocialButton startIcon={<GitHub />} href="https://github.com/sboesen2" target="_blank" rel="noopener noreferrer" color="primary" variant="outlined">
                GitHub
              </SocialButton>
            </Tooltip>
            <Tooltip title="Connect on LinkedIn" arrow TransitionComponent={Zoom}>
              <SocialButton startIcon={<LinkedIn />} href="https://www.linkedin.com/in/sam-boesen1" target="_blank" rel="noopener noreferrer" color="primary" variant="outlined">
                LinkedIn
              </SocialButton>
            </Tooltip>
            <Tooltip title="Send Email via Gmail" arrow TransitionComponent={Zoom}>
              <SocialButton
                startIcon={<Email />}
                href="https://mail.google.com/mail/?view=cm&fs=1&to=sam.boesen2@gmail.com"
                target="_blank"
                rel="noopener noreferrer"
                color="primary"
                variant="outlined"
              >
                Email
              </SocialButton>
            </Tooltip>
          </Box>
        </Fade>

        <Fade in={true} timeout={3500}>
          <Box mt={6} textAlign="center">
            <Button
              variant="outlined"
              color="primary"
              endIcon={<KeyboardArrowDown />}
              onClick={scrollToContent}
            >
              Learn More
            </Button>
          </Box>
        </Fade>

        <Divider sx={{ my: 6, width: '100%' }} />

        <Fade in={showContent} timeout={1000}>
          <Box id="author-content">
            <SectionTitle variant="h4">About Me</SectionTitle>
            <Typography paragraph align="center">
              I am a passionate researcher and developer focused on leveraging machine learning and data science to advance our understanding of Alzheimer's disease and other neurodegenerative illnesses. Im currently a rising junior at Arizona State University studying Biomedical Science. I have always been passionate about Biology and medicine but I recently discover the world of computation and its incessant need in the field.
              I then started learning about Data Science and all of the positive utilization it has in the Biomedical science space. The ability to do incredibly large, complex dataset analysis in a short time has unimaginable benefits and cost reduction for in vitro and in vivo research and development. However, there is a strong disconnect or a degree of scepticism from in silico research to wet lab biology which is completely understandable.
              Just because an experiment works on a computer simulation does not mean it will work in practice. This is the bridge of distrust that I hope to short or weaken with this project. I hope that all who use this tool can have some utilization for it or see the possibilities. I am more than open to collaboration or sharing my findings in much greater depth with any institutions or Research & Development firms that are interested.
              Please do not hesitate to reach out to me via Gmail or Linkedin.
            </Typography>

            <SectionTitle variant="h4">Project Overview</SectionTitle>
            <Typography paragraph align="center">
              The Alzheimer's Risk Prediction Tool is a culmination of extensive research and development in the field of genetic risk assessment for Alzheimer's disease. This project aims to provide researchers and healthcare professionals with a user-friendly interface to estimate genetic risk factors associated with Alzheimer's, based on cutting-edge machine learning algorithms and large-scale genetic studies.
              It is my firm belief that science at its core is a an industry that should be focused on finding innovative solutions to complex problems for the advancement of society.
              Too often are profits an ulterior motive for a product or treatment that is saving lives. That is why I have decided to make almost all of the code and Exploratory data analysis (EDA) that this website utilizes open source. If there is even one person that gain some piece of mind or can further their studies in the field of neurodegenerative diseases than all of this was worth it.
              I will not include the actual model itself with the tuned hyperparameters or the fully merged, cleaned data set as that is bad practice in the field of Big Data.
              However, all of the charts, tables, statistical analysis, and EDA that helped me create this model will be available. I invite anyone to analyze my notebooks and provide insights or comments about what they see. Anything that can advance the filed of Alzheimer's research and neurodegenerative diseases has the potential to save thousands of lives and improve the quality of life of even more.
            </Typography>

            <SectionTitle variant="h4">Key Features</SectionTitle>
            <Grid container spacing={3}>
              {['Advanced ML Algorithms', 'Comprehensive Genetic Analysis', 'User-Friendly Interface', 'Regular Updates'].map((feature) => (
                <Grid item xs={12} sm={6} key={feature}>
                  <FeatureCard elevation={3}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom color="primary">
                        {feature}
                      </Typography>
                      <Typography variant="body2">
                        {getFeatureDescription(feature)}
                      </Typography>
                    </CardContent>
                  </FeatureCard>
                </Grid>
              ))}
            </Grid>

            <SectionTitle variant="h4">Methodology</SectionTitle>
            <Typography paragraph>
              My risk prediction model utilizes a sophisticated machine learning approach, incorporating data from multiple reputable genetic databases and studies. The core of our methodology involves:
            </Typography>
            <List>
              {[
                'Analysis of key genetic markers associated with Alzheimer\'s risk',
                'Integration of data from large-scale genome-wide association studies (GWAS)',
                'Utilization of advanced statistical techniques to assess genetic risk factors',
                'Continuous refinement of the model based on the latest research findings'
              ].map((item, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <CheckCircleOutline color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={item} />
                </ListItem>
              ))}
            </List>

            <SectionTitle variant="h4">Future Directions</SectionTitle>
            <Typography paragraph>
              As I continue to develop and refine this tool, my future goals include:
            </Typography>
            <List>
              {[
                'Incorporating additional genetic markers as new research emerges',
                'Expanding the tool to assess risk for other neurodegenerative diseases',
                'Developing personalized prevention strategies based on genetic risk profiles',
                'Collaborating with research institutions to further validate and improve the model'
              ].map((item, index) => (
                <ListItem key={index}>
                  <ListItemIcon>
                    <CheckCircleOutline color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={item} />
                </ListItem>
              ))}
            </List>

            <SectionTitle variant="h4">Get Involved</SectionTitle>
            <Typography paragraph align="center">
              I'm always open to collaboration and feedback. If you're interested in contributing to this project or have suggestions for improvement, please don't hesitate to reach out. You can contribute to the project on <Link href="https://github.com/sboesen2/alzheimers-risk-prediction" target="_blank" rel="noopener noreferrer">GitHub</Link> or contact me directly.
            </Typography>

            <Box mt={6} textAlign="center">
              <Button
                variant="contained"
                color="primary"
                href="https://github.com/sboesen2/alzheimers-risk-prediction"
                target="_blank"
                rel="noopener noreferrer"
                size="large"
              >
                View Project on GitHub
              </Button>
            </Box>
          </Box>
        </Fade>
      </Box>
    </Container>
  );
}

function getFeatureDescription(feature) {
  const descriptions = {
    'Advanced ML Algorithms': 'Utilizes state-of-the-art machine learning models to analyze genetic data and predict Alzheimer\'s risk.',
    'Comprehensive Genetic Analysis': 'Incorporates a wide range of genetic markers associated with Alzheimer\'s disease for a thorough risk assessment.',
    'User-Friendly Interface': 'Designed with researchers and healthcare professionals in mind, offering an intuitive and easy-to-use platform.',
    'Regular Updates': 'Continuously updated to incorporate the latest findings in Alzheimer\'s genetic research.',
  };
  return descriptions[feature];
}

export default Author;