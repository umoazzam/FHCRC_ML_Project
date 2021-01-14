# FHCRC_ML_Project
 
## Title: Deep learning to identify ECG features associated with risk of stroke and stroke subtypes
### Setting: CHS cohort
### Investigators: Alison Fohner, Noah Simon, Bruce Psaty, Jessica Perry, Jen Brody, Susan Heckbert, Nathan Kutz, Will Longstreth, Barbara McKnight, Diya Sashidhar, Colleen Sitlani, Nona Sotoodehnia

### Background:
Strokes kill 10% of people worldwide, making them the second greatest killer after ischemic heart disease.1 Strokes are also a leading cause of disability. Currently, diagnosis requires clinical evaluation and brain imaging, which must occur promptly, as the specific type of stroke determines appropriate treatment.2-4 Electrocardiograms (ECGs) are easy to obtain and routinely are collected for other conditions. Identifying associations between ECG readings and risk of stroke would be valuable for both greater characterization of the etiology of stroke and its subtypes, and for improving prediction of future stroke risk.

Ischemic stroke accounts for approximately 80-85% of strokes. The subtypes of ischemic stroke include large vessel, small vessel, and cardioembolic, although many stroke do not fit into these three main categories.2 Most studies of ECGs and stroke focus on Atrial Fibrillation (AF) and the P-wave. AF is associated with a 5-fold increased risk of ischemic stroke and a 5-fold increased risk of recurrent stroke.5,6 AF is characterized in the absence of P-waves and an irregular ventricular rate on an ECG. However, only about ⅓ of ischemic stroke patients have ECGs indicating even intermittent AF, including after three years of continuous monitoring.7 

Independently of AF, the P-wave seems to be the most significant feature of ECGs associated with ischemic stroke.7-16 The most consistently and strongly associated is the P-wave terminal force in V1 (PTFV1), which is the amplitude below baseline, multiplied by the duration of the P-wave in the downward direction from the V1 lead. These values are measured automatically by Marquette ECG machines. Aberrations in the PTFV1 may indicate left atrial dilation or cardiopathy, which allows embolism formation.9,11,12,14,17,18   Alternative hypotheses for elevated PTFV1 include impaired interatrial conduction, atrial fibrosis, or elevated left atrial pressure15,19 PTFV1 is more strongly associated with stroke than it is with AF, suggesting that AF is not a mediator in this relationship.11 Hazard ratios of high PTFV1 for incident stroke are 1.2 per 1-standard deviation increase118 and 1.3 for PTFV1 > 4,000uV*ms compared to < 4,000 uV*ms10, with stronger associations for cryptogenic strokes and nonlacunar strokes.6,8,10,11,15 

Other features of an abnormal P-wave that have been associated with stroke include the P-wave axis,13 P-wave duration, P-wave area, and PR duration.16 In a study of young people (15-49 years) with stroke, the specific P-wave abnormalities were associated with different stroke types.15 Many studies report inability to account for subclinical AF as a limitation. 

Deep learning with ECGs could contribute to stroke research in several ways: 1) by clarifying associations between ECG features and the risk of incident ischemic stroke; 2) by differentiating subtypes of ischemic stroke apparent in ECG features; and 3) by illustrating how ECG features may improve predictions of ischemic stroke compared to AF alone. We propose applying deep learning to ECGs, including: 1) applying convolutional neural networks (CNN) and recurrent neural networks (RNNs) to ECGs to evaluate ECG traits as risk factors for incident stroke; and among an inception cohort of stroke survivors, to evaluate ECG traits as risk factors for recurrent ischemic stroke; and 2) determining how this method improves or overlaps with predictions of ischemic stroke based on PTFV1 and AF. 

### Study design:
This study will take place in the Cardiovascular Health Study (CHS), which includes up to 10 serial ECGs for each participant and adjudicated strokes.20,21 Over 1000 strokes are estimated in CHS. We will exclude participants with prevalent stroke at baseline. ECGs with poor quality (EPICARE score of 5), with atrial fibrillation, or with a pacemaker will be excluded. Initially a CNN will be built, with baseline ECG readings as input, to predict risk of incident ischemic stroke. This model will be trained with time-to-ischemic stroke as the outcome using a partial-likelihood-based loss. The data will be split randomly with 80% used to train the model and the remaining 20% to validate. Stroke prediction with CNN will be compared to predictions of ischemic stroke using AF, PTFV1, and clinical covariates. Predictions will be evaluated with log-hazard ratio from CNN, and improvement in time-dependent-AUC over baseline covariates alone, both including and not including AF and PTFV1. CNNs will be developed using raw ECG inputs, as well as inputs from a windowed-Fourier transform of ECG intensities. To leverage the serial ECGs, we will then employ a hierarchical RNN structure to serially connect outputs from CNNs associated with each patient’s individual ECGs. 

In secondary analysis, we will assemble an inception cohort of stroke survivors and apply CNN and RNNs to the first ECG after incident ischemic stroke and serial ECGs after incident stroke to predict recurrent ischemic stroke.

### CHS data needed: 
●	Serial ECGs throughout CHS study period, including raw data from 12-leads and Marquette readings
●	For each ECG: time from baseline, atrial fibrillation present, EPICARE quality score
●	All adjudicated incident ischemic stroke events and time from baseline
●	Prevalent ischemic stroke and atrial fibrillation at baseline
●	Pacemaker and time from baseline (self-report, Minnesota code 6-8, ICD-9 code V45.01 and V53.31 and procedure code 37.8*)
●	Clinical covariates at baseline (age, sex, race, hypertension, diabetes, HDL, LDL, education, BMI, triglycerides, SBP, anti-hypertensive meds, aspirin, blood thinners, and smoking)
●	Incident heart failure, atrial fibrillation, myocardial infarction, with time-to-event
 
### References:
1.	Donnan GA, Fisher M, Macleod M, Davis SM. Stroke. Lancet (London, England). 2008;371(9624):1612-1623.
2.	Hankey GJ. Stroke. Lancet (London, England). 2017;389(10069):641-654.
3.	Meschia JF, Bushnell C, Boden-Albala B, et al. Guidelines for the primary prevention of stroke: a statement for healthcare professionals from the American Heart Association/American Stroke Association. Stroke. 2014;45(12):3754-3832.
4.	Kernan WN, Ovbiagele B, Black HR, et al. Guidelines for the prevention of stroke in patients with stroke and transient ischemic attack: a guideline for healthcare professionals from the American Heart Association/American Stroke Association. Stroke. 2014;45(7):2160-2236.
5.	Kamel H, Johnson DR, Hegde M, et al. Detection of atrial fibrillation after stroke and the risk of recurrent stroke. Journal of stroke and cerebrovascular diseases : the official journal of National Stroke Association. 2012;21(8):726-731.
6.	Katsnelson M, Koch S, Rundek T. Stroke Prevention in Atrial Fibrillation. Journal of atrial fibrillation. 2010;3(3):279.
7.	Kamel H, Bartz TM, Elkind MSV, et al. Atrial Cardiopathy and the Risk of Ischemic Stroke in the CHS (Cardiovascular Health Study). Stroke. 2018;49(4):980-986.
8.	He J, Tse G, Korantzopoulos P, et al. P-Wave Indices and Risk of Ischemic Stroke: A Systematic Review and Meta-Analysis. Stroke. 2017;48(8):2066-2072.
9.	Kamel H, Bartz TM, Longstreth WT, Jr., et al. Association between left atrial abnormality on ECG and vascular brain injury on MRI in the Cardiovascular Health Study. Stroke. 2015;46(3):711-716.
10.	Kamel H, O'Neal WT, Okin PM, Loehr LR, Alonso A, Soliman EZ. Electrocardiographic left atrial abnormality and stroke subtype in the atherosclerosis risk in communities study. Annals of neurology. 2015;78(5):670-678.
11.	Kamel H, Soliman EZ, Heckbert SR, et al. P-wave morphology and the risk of incident ischemic stroke in the Multi-Ethnic Study of Atherosclerosis. Stroke. 2014;45(9):2786-2788.
12.	Kohsaka S, Sciacca RR, Sugioka K, Sacco RL, Homma S, Di Tullio MR. Electrocardiographic left atrial abnormalities and risk of ischemic stroke. Stroke. 2005;36(11):2481-2483.
13.	Maheshwari A, Norby FL, Soliman EZ, et al. Abnormal P-Wave Axis and Ischemic Stroke: The ARIC Study (Atherosclerosis Risk In Communities). Stroke. 2017;48(8):2060-2065.
14.	Okin PM, Kamel H, Kjeldsen SE, Devereux RB. Electrocardiographic left atrial abnormalities and risk of incident stroke in hypertensive patients with electrocardiographic left ventricular hypertrophy. Journal of hypertension. 2016;34(9):1831-1837.
15.	Pirinen J, Eranti A, Knekt P, et al. ECG markers associated with ischemic stroke at young age - a case-control study. Annals of medicine. 2017;49(7):562-568.
16.	Soliman EZ, Prineas RJ, Case LD, Zhang ZM, Goff DC, Jr. Ethnic distribution of ECG predictors of atrial fibrillation and its impact on understanding the ethnic distribution of ischemic stroke in the Atherosclerosis Risk in Communities (ARIC) study. Stroke. 2009;40(4):1204-1211.
17.	Kamel H, Okin PM, Longstreth WT, Jr., Elkind MS, Soliman EZ. Atrial cardiopathy: a broadened concept of left atrial thromboembolism beyond atrial fibrillation. Future cardiology. 2015;11(3):323-331.
18.	Yaghi S, Bartz TM, Kronmal R, et al. Left atrial diameter and vascular brain injury on MRI: The Cardiovascular Health Study. Neurology. 2018;91(13):e1237-e1244.
19.	Tandon K, Tirschwell D, Longstreth WT, Jr., Smith B, Akoum N. Embolic stroke of undetermined source correlates to atrial fibrosis without atrial fibrillation. Neurology. 2019;93(4):e381-e387.
20.	Ives DG, Fitzpatrick AL, Bild DE, et al. Surveillance and ascertainment of cardiovascular events. The Cardiovascular Health Study. Annals of epidemiology. 1995;5(4):278-285.
21.	Psaty BM, Delaney JA, Arnold AM, et al. Study of Cardiovascular Health Outcomes in the Era of Claims Data: The Cardiovascular Health Study. Circulation. 2016;133(2):156-164.

