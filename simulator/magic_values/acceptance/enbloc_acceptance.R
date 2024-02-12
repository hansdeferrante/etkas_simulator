
# How to discretize continuous variables for accepting organ offers in
# recipient-driven allocation

# Cuts to continuous variables
DISCRETIZATION_CUTS <- list2(
  #!!DELTA_WEIGHT := c(-25, -10, 0, 10, 25)#,
  !!DONOR_BMI := c(20, 25, 30),
  #!!MELD_LAB :=  c(10, 18, 25, 30)
)

# Which levels to use as reference level after cutting
DISCRETIZED_REF_LEVELS <- list2(
  !!DELTA_WEIGHT := 3,
  !!DONOR_AGE := 3,
  !!MELD_LAB := 1,
  !!DONOR_BMI := 2
)

# Raw adjustment variables
RAW_ADJ_VARS <- c(
  DONOR_AGE,
  D_WEIGHT,
  R_MATCH_AGE,
  R_WEIGHT,
  'recipient_country_chs'
)

# Locations for knots, if using splines.
KNOT_LOCATIONS <- list2(
  !!DONOR_AGE := c(4, 12, 25, 40, 55, 65),
  !!R_MATCH_AGE := c(4, 12, 25, 40, 55, 65),
  !!R_WEIGHT := c(25, 50, 75, 100),
  !!D_WEIGHT := c(25, 50, 75, 100)
)


BOUNDARY_KNOTS <- list2(
)


# Named vector to translante cont. var. to spline term.
SPLINE_TERMS <- imap(
  KNOT_LOCATIONS,
  ~ ifelse(
    .y %in% names(BOUNDARY_KNOTS),
    glue('ns({.y}, knots = c({paste0(.x, collapse=", ")}), Boundary.knots = c({chuck(BOUNDARY_KNOTS, .y, 1)}, {chuck(BOUNDARY_KNOTS, .y, 2)}))'),
    glue('ns({.y}, knots = c({paste0(.x, collapse=", ")}))')
  )
)

# Adjustment variables with all spline terms.
SPLINE_ADJ <- recode(
  RAW_ADJ_VARS,
  !!!SPLINE_TERMS
)

# Named vector which adjustment variables to use when using discretized info.
DISCRETIZED_VARS <- list2(
  !!R_WEIGHT := c(
    R_WEIGHT,
    glue::glue('f_under_50({R_WEIGHT})'),
    glue::glue('f_over_90({R_WEIGHT})')
  ) |> paste0(collapse = " + "),
  !!DONOR_AGE := c(
    glue::glue('f_under_16({DONOR_AGE})'),
    glue::glue('f_over_50({DONOR_AGE})'),
    c(DONOR_AGE),
    glue::glue('f_over_65({DONOR_AGE})')
  ) |> paste0(collapse = " + "),
  !!R_MATCH_AGE := c(
    R_MATCH_AGE,
    glue::glue('f_under_8({R_MATCH_AGE})'),
    glue::glue('f_under_40({R_MATCH_AGE})')
  ) |> paste0(collapse = ' + '),
  !!D_WEIGHT := c(
    D_WEIGHT,
    glue::glue('f_under_50({D_WEIGHT})'),
    glue::glue('f_under_75({D_WEIGHT})'),
    glue::glue('f_over_100({D_WEIGHT})')
  ) |> paste0(collapse = ' + ')
)

# All adjustment variables discretized.
DISCR_ADJ <- recode(
  RAW_ADJ_VARS,
  !!!DISCRETIZED_VARS
) %>%
  recode(
    !!R_MATCH_AGE := R_MATCH_AGE_P10,
    !!MELD_LAB := MELD_GT_18
  )

f_over_c <- function(c) {function(x) pmax(0, x-c)}
f_under_c <- function(c) {function(x) pmax(0, c-x)}
f_over_50 <- f_over_c(50)
f_over_65 <- f_over_c(65)
f_under_8 <- f_under_c(8)
f_under_40 <- f_under_c(40)
f_under_20 <- f_under_c(20)
f_under_16 <- f_under_c(16)
f_over_90 <- f_over_c(90)
f_under_50 <- f_under_c(50)
f_under_75 <- f_under_c(75)
f_over_100 <- f_over_c(100)


