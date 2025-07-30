# Inter-Annotator Agreement (IAA) Analysis Report
## Challenge #1 Deliverable: Evaluating Annotation Consistency

### Executive Summary

This report presents the inter-annotator agreement analysis for the antisemitism detection project, evaluating the consistency between two annotators who classified 100 tweets according to the IHRA Working Definition of Antisemitism (IHRA-WDA).

**Key Findings:**
- **Cohen's Kappa: 0.118** (Slight agreement)
- **Krippendorff's Alpha: -0.169** (Reliability insufficient)
- **Agreement Level: Low** - Significant disagreement between annotators
- **Disagreements: 56 out of 99 valid comparisons** (56.6% disagreement rate)


### Methodology

#### Dataset and Annotators
- **Dataset**: Sample4 dataset containing 100 tweets
- **Annotators**: 
  - Annotator 1: danielm
  - Annotator 2: dshink
- **Annotation Framework**: IHRA Working Definition of Antisemitism (IHRA-WDA)
- **Classification Categories**: 
  - Confident antisemitic
  - Probably antisemitic
  - Probably not antisemitic
  - Confident not antisemitic
  - I don't know

#### Data Collection Process
1. **Data Scraping**: Tweets were collected from social media platforms using the Bright Data scraping tool
2. **Annotation Guidelines**: Annotators followed the IHRA-WDA framework with specific instructions for classification
3. **Double Annotation**: Each tweet was independently classified by both annotators
4. **Quality Control**: Uncertain classifications ("I don't know") were excluded from agreement calculations

### Results

#### Annotation Distribution

**Annotator 1 (danielm):**
- Confident not antisemitic: 69 tweets (69%)
- Probably not antisemitic: 18 tweets (18%)
- Confident antisemitic: 6 tweets (6%)
- Probably antisemitic: 5 tweets (5%)
- I don't know: 1 tweet (1%)
- Mixed classification: 1 tweet (1%)

**Annotator 2 (dshink):**
- Confident antisemitic: 57 tweets (57%)
- Confident not antisemitic: 20 tweets (20%)
- Probably not antisemitic: 11 tweets (11%)
- Probably antisemitic: 9 tweets (9%)
- Mixed classification: 3 tweets (3%)

#### Agreement Metrics

**Cohen's Kappa: 0.118**
- **Interpretation**: Slight agreement
- **Range**: -1 to +1 (where +1 is perfect agreement)
- **Assessment**: Agreement is only slightly better than chance

**Krippendorff's Alpha: -0.169**
- **Interpretation**: Reliability insufficient
- **Range**: -1 to +1 (where +1 is perfect agreement)
- **Assessment**: Agreement is worse than chance, indicating systematic disagreement

#### Disagreement Analysis

**Total Disagreements: 56 out of 99 valid comparisons (56.6%)**

**Sample Disagreement Cases:**
1. Tweet ID: 1809140786673356904
   - Annotator 1: Confident not antisemitic
   - Annotator 2: Probably antisemitic

2. Tweet ID: 1933644653045428598
   - Annotator 1: Confident not antisemitic
   - Annotator 2: Probably antisemitic, Confident antisemitic

3. Tweet ID: 1947042844939899178
   - Annotator 1: Probably not antisemitic
   - Annotator 2: I don't know, Confident antisemitic

### Tweet analysis
**Most common words in tweets labled antisemitic by at least 1 annotator:**
 1. israel          -  22 occurrences
 2. gaza            -  19 occurrences
 3. palestinians    -  11 occurrences
 5. people          -  10 occurrences
 7. israeli         -   7 occurrences
 9. genocide        -   7 occurrences
10. palestine       -   6 occurrences
11. against         -   6 occurrences
14. aid             -   6 occurrences
17. stop            -   5 occurrences
18. october         -   5 occurrences

### Discussion

#### Interpretation of Results

The low agreement scores indicate significant challenges in consistently applying the IHRA-WDA framework:

1. **Systematic Bias**: Annotator 2 (dshink) classified 57% of tweets as antisemitic, while Annotator 1 (danielm) classified only 12% as antisemitic. This suggests different interpretations of the IHRA-WDA criteria.

2. **Framework Ambiguity**: The IHRA-WDA, while comprehensive, may contain subjective elements that lead to different interpretations among annotators.

3. **Context Sensitivity**: Antisemitism detection often requires nuanced understanding of context, cultural references, and intent, which may vary between annotators.

#### Implications for the Project

1. **Annotation Quality**: The low agreement suggests the need for additional training and clearer guidelines for annotators.

2. **Model Reliability**: Any machine learning model trained on this data may inherit the inconsistencies present in the annotations.

3. **Validation Strategy**: The project should implement additional validation steps and potentially involve more annotators for consensus.

### Recommendations

#### Short-term Actions
1. **Annotator Training**: Provide additional training on IHRA-WDA application with clear examples
2. **Guideline Refinement**: Develop more specific criteria for edge cases and ambiguous content
3. **Pilot Study**: Conduct a smaller pilot with revised guidelines before full annotation

#### Long-term Improvements
1. **Multi-Annotator Consensus**: Implement a three-annotator system with majority voting
2. **Expert Review**: Include domain experts in the annotation process
3. **Continuous Monitoring**: Regular IAA assessments during the annotation process

### Conclusion

The inter-annotator agreement analysis reveals significant challenges in consistently applying the IHRA-WDA framework for antisemitism detection. The low agreement scores (Cohen's Kappa: 0.118, Krippendorff's Alpha: -0.169) indicate that the current annotation process requires substantial improvement before it can reliably support machine learning model development.

**Bonus Points Assessment**: The current agreement level (slight agreement) does not meet the threshold for bonus points, which typically require moderate or higher agreement (Cohen's Kappa ≥ 0.4).

### Technical Appendix

#### Statistical Details
- **Valid Comparisons**: 99 out of 100 tweets (1 excluded due to "I don't know" classification)
- **Confusion Matrix**: Available in visualization file (iaa_agreement_matrix.png)
- **Statistical Software**: Python with scikit-learn and krippendorff libraries

#### Data Processing
- **Preprocessing**: Standardized classification categories, excluded uncertain responses
- **Binary Classification**: Converted to binary (antisemitic = 1, not antisemitic = 0) for agreement calculations
- **Missing Data**: Handled by exclusion from agreement calculations

---

*Report generated on: July 30, 2025*
*Analysis performed by: IAA Analysis Script*
*Dataset: Sample4 (100 tweets)* 