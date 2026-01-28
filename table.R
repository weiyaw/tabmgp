library(tidyverse)
library(glue)
library(kableExtra)


synthetic_order <- c(
  "regression-standard",
  "regression-t-5",
  "regression-t-4",
  "regression-t-3",
  "regression-dependent-0.25-0.5",
  "regression-dependent-0.05-0.25",
  "regression-dependent-0.01-0.1",
  "classification-standard",
  "classification-gmm-0",
  "classification-gmm--1",
  "classification-gmm--2"
)

style_joint_table <- function(tbl) {
  ## tbl: a data frame with these columns exactly: setup, post_name, rate, size
  ## old_options <- options(knitr.kable.NA = '')
  ## on.exit(options(old_options)) # Clean up options after function runs

  expected_methods <- c("TabMGP", "BB", "Copula", "Bayes", "Asymptotic")

  tbl_wide <- tbl |>
    pivot_wider(
      names_from = post_name,
      values_from = c(rate, size),
      names_glue = "{post_name}_{.value}",
      names_vary = "slowest"
    )

  # double check if the column names are correct to prevent mislabeling
  expected_cols <- c("setup", unlist(lapply(expected_methods, function(x) paste0(x, c("_rate", "_size")))))
  stopifnot(all(colnames(tbl_wide) == expected_cols))

  tbl_wide |>
    kable(
      format = "latex",
      booktabs = TRUE,
      escape = FALSE,
      linesep = "",
      digits = c(0, rep(c(2, 2), 5)), # 0 for Setup, then 2 for every Rate and Size
      format.args = list(nsmall = 2), # Ensures NA is handled correctly
      align = "lcccccccccc",
      col.names = c("Setup", rep(c("Rate", "Size"), 5)),
      table.envir = "table*"
    ) |>
    add_header_above(c(" " = 1, "TabMGP" = 2, "BB" = 2, "Copula" = 2, "Bayes" = 2, "Asymptotic" = 2))
}

## filter out useless rows and columns, and rename posterior
joint_coverage <- read_csv("table/joint-coverage.csv") |>
  filter(max_T, post_name %in% c("tabpfn", "bb", "copula", "gibbs-eb", "clt"), ideal_rate == 0.95) |>
  select(setup = data, functional, post_name, rate, size = post_cov_trace_median, dim_theta, n_train = training_set_size) |>
  mutate(
    np_ratio = zapsmall(n_train / dim_theta, 2),
    post_name = factor(post_name,
      levels = c("tabpfn", "bb", "copula", "gibbs-eb", "clt"),
      labels = c("TabMGP", "BB", "Copula", "Bayes", "Asymptotic")
    )
  ) |>
  arrange(desc(np_ratio), post_name) %>%
  group_by(setup) |>
  mutate(
    covered = rate >= 0.95 | rate == max(rate, na.rm = TRUE),
    min_covered = min(size[covered]),
    bold = if_else(covered, size == min_covered, FALSE),
    across(c(rate, size), ~ ifelse(bold, paste0("\\textbf{", format(round(.x, 2), nsmall = 2), "}"), format(round(.x, 2), nsmall = 2)))
  )



syn_reg <- joint_coverage |>
  filter(str_detect(setup, "regression-"), functional == "likelihood-gaussian") |>
  mutate(setup_factor = factor(setup, levels = synthetic_order)) |>
  arrange(setup_factor) |>
  mutate(setup = case_when(
    setup == "regression-standard" ~ "$\\sN(0, 1)$",
    setup == "regression-t-5" ~ "$t_5$",
    setup == "regression-t-4" ~ "$t_4$",
    setup == "regression-t-3" ~ "$t_3$",
    setup == "regression-dependent-0.25-0.5" ~ "$s_1$",
    setup == "regression-dependent-0.05-0.25" ~ "$s_2$",
    setup == "regression-dependent-0.01-0.1" ~ "$s_3$",
    TRUE ~ setup
  ))

real_reg <- joint_coverage |>
  filter(str_starts(setup, "regression", negate = TRUE), functional == "likelihood-gaussian") |>
  arrange(functional, desc(np_ratio))


bind_rows(syn_reg, real_reg) |>
  select(setup, post_name, rate, size) |>
  style_joint_table() |>
  row_spec(c(4, 7), hline_after = TRUE)




# table with classification data
syn_class <- joint_coverage |>
  filter(
    str_detect(setup, "classification-"),
    str_starts(functional, "likelihood"),
  ) |>
  mutate(setup_factor = factor(setup, levels = synthetic_order)) |>
  arrange(setup_factor) |>
  mutate(setup = case_when(
    setup == "classification-standard" ~ "Logistic",
    setup == "classification-gmm-0" ~ "GMM$(0)$",
    setup == "classification-gmm--1" ~ "GMM$(-1)$",
    setup == "classification-gmm--2" ~ "GMM$(-2)$",
    TRUE ~ setup
  ))

real_class <- joint_coverage |>
  filter(
    str_starts(setup, "classification", negate = TRUE),
    functional %in% c("likelihood-binary", "likelihood-multiclass"),
  ) |>
  arrange(functional, desc(np_ratio))

bind_rows(syn_class, real_class) |>
  select(setup, post_name, rate, size) |>
  style_joint_table() |>
  row_spec(c(4, 12), hline_after = TRUE)



## SETUP INFO
data_info <- read_csv("data_info.csv") |>
  rename(n_train = training_size, data = name) |>
  select(-c(date, x_dtype)) |>
  mutate(np_ratio = zapsmall(n_train / dim_theta, 2))


data_info |>
  filter(!str_detect(data, "(regression|classification)-")) |>
  mutate(
    across(c(n_train:n_categorical_features, dim_theta), as.integer),
    openml_id = ifelse(is.na(openml_id), NA, glue("OpenML {openml_id}")),
    uci_id = ifelse(is.na(uci_id), NA, glue("UCI {uci_id}")),
    data_id = coalesce(openml_id, uci_id)
  ) |>
  group_by(n_classes_in_y) |>
  arrange(desc(np_ratio), .by_group = TRUE) |>
  ungroup() |>
  select("$z_{1:n}$" = data, data_id, n_train, dim_theta, np_ratio, population_size, n_classes_in_y:n_categorical_features, -n_features) |>
  rename("Data ID" = data_id, "$n$" = n_train, "$p$" = dim_theta, "$n/p$" = np_ratio, "Population size" = population_size, "\\# classes" = n_classes_in_y, "\\# cont. features" = n_continuous_features, "\\# cat. features" = n_categorical_features) |>
  xtable() |>
  print(
    include.rownames = FALSE,
    sanitize.text.function = identity,
    add.to.row = list(pos = list(9, 9 + 8), command = c("\\midrule ", "\\midrule ")),
    booktabs = TRUE,
  )

data_info |>
  filter(!str_detect(data, "(regression|classification)-")) |>
  mutate(functional_feature_drop = str_remove_all(functional_feature_drop, "[\\[\\]']")) |>
  group_by(n_classes_in_y) |>
  arrange(desc(np_ratio), .by_group = TRUE) |>
  ungroup() |>
  select("$z_{1:n}$" = data, "Target Name" = target_name, "Dropped features in the functional" = functional_feature_drop) |>
  xtable() |>
  print(
    include.rownames = FALSE,
    add.to.row = list(pos = list(9, 9 + 8), command = c("\\midrule ", "\\midrule ")),
    booktabs = TRUE,
  )



## MARGINAL CI
marginal_coverage <- read_csv("table/marginal-coverage.csv") |>
  filter(max_T, post_name %in% c("tabpfn", "bb", "copula", "gibbs-eb", "clt"), ideal_rate == 0.95) |>
  select(setup = data, theta_name, post_name, rate, size = median_width) |>
  mutate(
    post_name = factor(post_name,
      levels = c("tabpfn", "bb", "copula", "gibbs-eb", "clt"),
      labels = c("TabMGP", "BB", "Copula", "Bayes", "Asymptotic")
    )
  ) |>
  group_by(setup, theta_name) |>
  arrange(setup, theta_name, post_name) |>
  mutate(
    covered = rate >= 0.95 | rate == max(rate, na.rm = TRUE),
    min_covered = min(size[covered]),
    bold = if_else(covered, size == min_covered, FALSE),
    across(c(rate, size), ~ ifelse(bold, paste0("\\textbf{", format(round(.x, 2), nsmall = 2), "}"), format(round(.x, 2), nsmall = 2)))
  ) |>
  ungroup() |>
  select(setup, theta_name, post_name, rate, size)


style_marginal_table <- function(tbl) {
  present_methods <- tbl |>
    pull(post_name) |>
    unique() |>
    sort() |>
    as.character()

  n_methods <- length(present_methods)

  tbl_wide <- tbl |>
    mutate(theta_name = str_replace_all(theta_name, "_", "\\\\_")) |>
    pivot_wider(
      names_from = post_name, values_from = c(rate, size),
      names_glue = "{post_name}_{.value}",
      names_vary = "slowest"
    )

  # double check if the column names are correct to prevent mislabeling
  expected_cols <- c("setup", "theta_name", unlist(lapply(present_methods, function(x) paste0(x, c("_rate", "_size")))))
  stopifnot(all(colnames(tbl_wide) == expected_cols))

  tbl_wide |>
    kable(
      format = "latex",
      booktabs = TRUE,
      escape = FALSE,
      linesep = "",
      align = paste0("ll", strrep("c", n_methods * 2)),
      col.names = c("Setup", "Feature Name", rep(c("Rate", "Size"), n_methods)),
    ) |>
    add_header_above(c(" " = 2, setNames(rep(2, n_methods), present_methods))) |>
    collapse_rows(columns = 1, latex_hline = "major", valign = "middle") |>
    kable_styling(position = "center")
}

marginal_coverage |>
  filter(str_detect(setup, "regression-(standard|t)")) |>
  arrange(factor(setup, levels = synthetic_order)) |>
  mutate(setup = case_when(
    setup == "regression-standard" ~ "$N(0, 1)$",
    setup == "regression-t-5" ~ "$t_5$",
    setup == "regression-t-4" ~ "$t_4$",
    setup == "regression-t-3" ~ "$t_3$",
    TRUE ~ setup
  )) |>
  style_marginal_table()


marginal_coverage |>
  filter(str_detect(setup, "regression-(dependent)")) |>
  arrange(factor(setup, levels = synthetic_order)) |>
  mutate(setup = case_when(
    setup == "regression-dependent-0.25-0.5" ~ "$s_1$",
    setup == "regression-dependent-0.05-0.25" ~ "$s_2$",
    setup == "regression-dependent-0.01-0.1" ~ "$s_3$",
    TRUE ~ setup
  )) |>
  style_marginal_table()


marginal_coverage |>
  filter(str_detect(setup, "classification")) |>
  arrange(factor(setup, levels = synthetic_order)) |>
  mutate(setup = case_when(
    setup == "classification-standard" ~ "Logistic",
    setup == "classification-gmm-0" ~ "GMM$(0)$",
    setup == "classification-gmm--1" ~ "GMM$(-1)$",
    setup == "classification-gmm--2" ~ "GMM$(-2)$",
    TRUE ~ setup
  )) |>
  style_marginal_table()



real_regression_1 <- c("concrete", "quake", "airfoil", "energy", "fish", "kin8nm")
marginal_coverage |>
  filter(setup %in% real_regression_1) |>
  arrange(factor(setup, levels = real_regression_1)) |>
  style_marginal_table()


real_regression_2 <- c("auction", "grid", "abalone")
marginal_coverage |>
  filter(setup %in% real_regression_2) |>
  arrange(factor(setup, levels = real_regression_2)) |>
  style_marginal_table()


real_classification_1 <- c("rice", "sepsis", "banknote", "mozilla", "skin", "blood", "phoneme", "telescope")
marginal_coverage |>
  filter(setup %in% real_classification_1) |>
  arrange(factor(setup, levels = real_classification_1)) |>
  style_marginal_table()

real_classification_2 <- c("yeast")
marginal_coverage |>
  filter(setup %in% real_classification_2) |>
  arrange(factor(setup, levels = real_classification_2)) |>
  style_marginal_table()

real_classification_3 <- c("wine")
marginal_coverage |>
  filter(setup %in% real_classification_3) |>
  arrange(factor(setup, levels = real_classification_3)) |>
  style_marginal_table()


read_csv("table/marginal-coverage.csv") |>
  filter(max_T, post_name %in% c("tabpfn", "bb", "copula", "gibbs-eb", "clt"), ideal_rate == 0.95) |>
  select(setup = data, theta_name, post_name, rate, size = median_width) |>
  mutate(
    post_name = factor(post_name,
      levels = c("tabpfn", "bb", "copula", "gibbs-eb", "clt"),
      labels = c("TabMGP", "BB", "Copula", "Bayes", "Asymptotic")
    )
  ) |>
  group_by(setup, theta_name) |>
  arrange(setup, theta_name, post_name) |>
  mutate(
    covered = rate >= 0.95 | rate == max(rate, na.rm = TRUE),
    min_covered = min(size[covered]),
    bold = if_else(covered, size == min_covered, FALSE),
    across(c(rate, size), ~ ifelse(bold, paste0("\\textbf{", format(round(.x, 2), nsmall = 2), "}"), format(round(.x, 2), nsmall = 2)))
  ) |>
  ungroup() |>
  filter(bold) |>
  count(post_name, bold)
