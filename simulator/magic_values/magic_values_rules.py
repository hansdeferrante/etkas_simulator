# Country names
AUSTRIA = 'Austria'
BELGIUM = 'Belgium'
CROATIA = 'Croatia'
GERMANY = 'Germany'
HUNGARY = 'Hungary'
LUXEMBOURG = 'Luxembourg'
NETHERLANDS = 'Netherlands'
SLOVENIA = 'Slovenia'
OTHER = 'Other'

# Center codes
CENTER_INNSBRUCK = 'AIBTP'
CENTER_VIENNA = 'AWGTP'
CENTER_GRAZ = 'AGATP'
CENTER_UPPERAUSTRIA = 'AOETP'

# Immunization statuses
T = 'T'
HU = 'HU'
NT = 'NT'
HI = 'HI'
I = 'I'

# Other statuses
FU = 'FU'
R = 'R'
D = 'D'

# Blood group
A = 'A'
B = 'B'
AB = 'AB'
O = 'O'

# HLA types
HLA_A = 'hla_a'
HLA_B = 'hla_b'
HLA_DR = 'hla_dr'
HLA_DQA = 'hla_dqa'
HLA_DQB = 'hla_dqb'
HLA_LOCI = (HLA_A, HLA_B, HLA_DR, HLA_DQA, HLA_DQB)


PUBLICS = tuple('Cw' + str(i) for i in range(20)) + ('Bw6', 'Bw4')
PUBLICS = set(tuple(x.upper() for x in PUBLICS))

# Unsplittables
DR3 = 'DR3'
DR17 = 'DR17'
DR18 = 'DR18'

IMPORT = 'import'
EXPORT = 'export'
NATIONAL = 'national'
REGIONAL = 'regional'

BROADS = 'broads'
SPLITS = 'splits'

MMB_HLA_A = 'mmb_hla_a'
MMB_HLA_B = 'mmb_hla_b'
MMB_HLA_DR = 'mmb_hla_dr'

MMS_HLA_A = 'mms_hla_a'
MMS_HLA_B = 'mms_hla_b'
MMS_HLA_DR = 'mms_hla_dr'

NHS_MISMATCH_LEVEL = 'nhs_mm_level'
ET_MISMATCH_LEVEL = 'et_mm_level'

FH = 'fh'
DR_plus = 'dr+'
DR0 = 'dr0',
DR1 = 'dr1',
DR2 = 'dr2'


PED = 'PED'
PRF = 'PRF'
DIAG = 'DIAG'
HLA = 'HLA'
UNACC = 'UNACC'
URG = 'URG'
AM = 'AM'
DIAL = 'DIAL'
PRA = 'PRA'

ESP = 'ESP'
ETKAS = 'ETKAS'
