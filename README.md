# loan_application_segmentation
segmentation of loan application
segmentation is of two types objective segmentation and subjective segmentation
objective segmentation -----where there is no dependent variable ------------we have to use clustering to solve this
subjective segmentation --------------------here we have a dependent variable-------------we should make use of classification /regression depending on variable
our eg we are using classification
here our segmentation is based on likelihood of customer accepting our offer of loan
the solution provided in the project is not to be consumed as a model but it will be segment of profiles to be targeted for increasing the business and profitability


features/variables used in project along with their descriptions

ID	Unique ID (cant be used for predictions)
Gender	Sex of the applicant
DOB	Date of Birth of the applicant
Lead Creation Date	Date on which Lead was created
City Code	Anonymised Code for the City
City Category	Anonymised City Feature
Employer Code	Anonymised Code for the Employer
Employer_Category1	Anonymised Employer Feature
Employer_Category2	Anonymised Employer Feature
Monthly Income	Monthly Income in Dollars
Customer Existing Primary Bank Code	Anonymised Customer Bank Code


Variable	Description
Primary Bank Type	Anonymised Bank Feature
Contacted	Contact Verified (Y/N)
Source	Categorical Variable representing source of lead
Source Category	Type of Source
Existing EMI	EMI of Existing Loans in Dollars
Loan Amount	Loan Amount Requested
Loan Period	Loan Period (Years)
Interest Rate	Interest Rate of Submitted Loan Amount
EMI	EMI of Requested Loan Amount in dollars
Var1	Categorical variable with multiple levels
Approved	(Target) Whether a loan is Approved or not (0/1)

# Concluding Notes
# The business team can notify the Loan Approval team to essentially target the applications which fall under 'Top 3' 
# followed by Mid 2 deciles

# Probability Cutoff to be applied
# Phase 1 - Focus on Prob value >=0.031460 [ Near about 10457 loan applications]
# Phase 2 - Focus on Prob value >=0.016718 and <=0.031460 [ Near about 6971 Loan applications ]


