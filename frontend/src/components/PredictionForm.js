import React, { useState } from 'react';

function PredictionForm({ onSubmit }) {
  const [formData, setFormData] = useState({
    apoeGenotype: '',
    tomm40Length: '',
    cluGenotype: '',
    picalmGenotype: '',
    cr1Genotype: ''
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        name="apoeGenotype"
        value={formData.apoeGenotype}
        onChange={handleChange}
        placeholder="APOE Genotype"
      />
      <input
        type="text"
        name="tomm40Length"
        value={formData.tomm40Length}
        onChange={handleChange}
        placeholder="TOMM40 Length"
      />
      <input
        type="text"
        name="cluGenotype"
        value={formData.cluGenotype}
        onChange={handleChange}
        placeholder="CLU Genotype"
      />
      <input
        type="text"
        name="picalmGenotype"
        value={formData.picalmGenotype}
        onChange={handleChange}
        placeholder="PICALM Genotype"
      />
      <input
        type="text"
        name="cr1Genotype"
        value={formData.cr1Genotype}
        onChange={handleChange}
        placeholder="CR1 Genotype"
      />
      <button type="submit">Predict Risk</button>
    </form>
  );
}

export default PredictionForm;