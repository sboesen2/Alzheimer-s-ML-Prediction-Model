import React, { Suspense } from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, AppBar, Toolbar, Typography, Container, Button, Box, CircularProgress } from '@mui/material';
import Home from './components/Home';
import Predict from './components/Predict';
import About from './components/About';
import FAQ from './components/FAQ';
import Footer from './components/Footer';
import DataSources from './components/DataSources';
import Author from './components/Author'; // Import the new Author component

// Create a custom dark theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#57c5b6', // Teal
    },
    secondary: {
      main: '#ff69b4', // Pink
    },
    background: {
      default: '#0a192f', // Dark blue
      paper: '#172a45', // Slightly lighter blue for components
    },
    text: {
      primary: '#ffffff',
      secondary: '#8892b0',
    },
  },
  typography: {
    fontFamily: '"Open Sans", "Helvetica", "Arial", sans-serif',
    h1: {
      fontFamily: '"Montserrat", "Helvetica", "Arial", sans-serif',
      fontWeight: 700,
      color: '#57c5b6',
    },
    h2: {
      fontFamily: '"Montserrat", "Helvetica", "Arial", sans-serif',
      fontWeight: 600,
      color: '#57c5b6',
    },
    h3: {
      fontFamily: '"Montserrat", "Helvetica", "Arial", sans-serif',
      fontWeight: 600,
      color: '#57c5b6',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 30,
          textTransform: 'none',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#172a45',
        },
      },
    },
  },
});

function ScrollToTop() {
  const { pathname } = useLocation();

  React.useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.log(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }

    return this.props.children;
  }
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <ScrollToTop />
        <Box display="flex" flexDirection="column" minHeight="100vh" bgcolor="background.default">
          <AppBar position="static" elevation={0}>
            <Toolbar>
              <Typography variant="h6" component={Link} to="/" style={{ flexGrow: 1, textDecoration: 'none', color: 'inherit' }}>
                Alzheimer's Risk Prediction
              </Typography>
              <Button color="inherit" component={Link} to="/">Home</Button>
              <Button color="inherit" component={Link} to="/predict">Predict</Button>
              <Button color="inherit" component={Link} to="/about">About</Button>
              <Button color="inherit" component={Link} to="/author">Author</Button> {/* New Author button */}
              <Button color="inherit" component={Link} to="/faq">FAQ</Button>
              <Button color="inherit" component={Link} to="/data-sources">Data Sources</Button>
            </Toolbar>
          </AppBar>
          <Container component="main" maxWidth="lg" style={{ marginTop: '2rem', flexGrow: 1 }}>
            <ErrorBoundary>
              <Suspense fallback={<CircularProgress />}>
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/predict" element={<Predict />} />
                  <Route path="/about" element={<About />} />
                  <Route path="/author" element={<Author />} /> {/* New Author route */}
                  <Route path="/faq" element={<FAQ />} />
                  <Route path="/data-sources" element={<DataSources />} />
                </Routes>
              </Suspense>
            </ErrorBoundary>
          </Container>
          <Footer />
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;