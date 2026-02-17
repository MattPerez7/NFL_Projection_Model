import pandas as pd
import numpy as np
from src.config import *

# Create utility functions

def _shifted_rolling(series, window = 3, min_period = 1):
    return series.shift(1).rolling(window, min_period).mean()

def _shifted_expanding_mean(series):
    return series.shift(1).expanding().mean()

# Create stat generator functions

# Merge needed pbp data for off pos groups missing metrics (not in df)
def calculate_off_adv(df_pbp):
    valid_plays = df_pbp[
        (df_pbp['play_type'].isin(['pass', 'run'])) &
        (df_pbp['aborted_play'] == 0)
    ].copy()

    # QB 
    valid_plays['redzone_completion'] = (((valid_plays['complete_pass'] == 1) & (valid_plays['yardline_100'] <= 20)))
    valid_plays['greenzone_completion'] = (((valid_plays['complete_pass'] == 1) & (valid_plays['yardline_100'] <= 5)))
    valid_plays['redzone_attempt'] = (((valid_plays['pass_attempt'] == 1) & (valid_plays['yardline_100'] <= 20)))
    valid_plays['greenzone_attempt'] = (((valid_plays['pass_attempt'] == 1) & (valid_plays['yardline_100'] <= 5)))

    qb_stats = valid_plays.groupby(['passer_id', 'season', 'week']).agg({
        'qb_dropback': 'sum',
        'cpoe': 'mean',
        'epa': 'mean',
        'redzone_completion': 'sum',
        'redzone_attempt': 'sum',
        'greenzone_completion': 'sum',
        'greenzone_attempt': 'sum'
    }).reset_index().rename(columns = {'passer_id': 'player_id', 'epa': 'epa_per_play'})

    # HVTs for Skill Groups
    valid_plays['is_hvt'] = ((valid_plays['pass_attempt'] == 1) | 
                             ((valid_plays['rush_attempt'] == 1) & (valid_plays['yardline_100'] <= 10)))
    valid_plays['redzone_carries'] = (((valid_plays['rush_attempt'] == 1) & (valid_plays['yardline_100'] <= 20)))
    valid_plays['greenzone_carries'] = (((valid_plays['rush_attempt'] == 1) & (valid_plays['yardline_100'] <= 5)))
    valid_plays['redzone_receptions'] = (((valid_plays['complete_pass'] == 1) & (valid_plays['yardline_100'] <= 20)))
    valid_plays['greenzone_receptions'] = (((valid_plays['complete_pass'] == 1) & (valid_plays['yardline_100'] <= 5)))

    # Success Rate: % of plays with positive EPA
    valid_plays['is_success'] = (valid_plays['epa'] > 0).astype(int)

    hvt_stats = valid_plays.groupby(['fantasy_player_id', 'season', 'week']).agg({
        'is_hvt': 'sum',
        'redzone_carries': 'sum',
        'greenzone_carries': 'sum',
        'redzone_receptions': 'sum',
        'greenzone_receptions': 'sum',
        'is_success': 'mean'
    }).reset_index().rename(columns = {'fantasy_player_id': 'player_id', 'is_success': 'success_rate'})

    return qb_stats, hvt_stats

def calculate_team_snaps_context(pbp):
    # Team Volume
    team_vol = pbp.groupby(['posteam', 'season', 'week']).agg({
        'play_id': 'count',         # Total Snaps
        'pass_attempt': 'sum',      # Total Pass Attempts
        'rush_attempt': 'sum'       # Total Rush Attempts
    }).reset_index().rename(columns = {
        'play_id': 'team_off_snaps',
        'pass_attempt': 'team_pass_attempts',
        'rush_attempt': 'team_rush_attempts'
    })

    # Neutral Pass Rate (Score within 7 points)
    neutral_plays = pbp[pbp['score_differential'].between(-7, 7)]

    # Calculate Pass Rate
    npr = neutral_plays.groupby(['posteam', 'season', 'week'])['pass_attempt'].mean().reset_index()
    npr = npr.rename(columns = {'pass_attempt': 'neutral_pass_rate'})

    # Merge
    team_context = team_vol.merge(npr, on = ['posteam', 'season', 'week'], how = 'left')

    # Fill NaN (no neutral script) with team's raw pass rate
    team_context['neutral_pass_rate'] = team_context['neutral_pass_rate'].fillna(
        team_context['team_pass_attempts'] / team_context['team_off_snaps']
    )

    return team_context

# Add season metrics for rolling and base
def add_season_metrics(df, metrics, group_cols = None):
    df = df.copy()

    if group_cols is None:
        # Create group keys
        if 'player_id' in df.columns:
            group_cols = ['player_id']
        elif 'opponent_team' in df.columns:
            group_cols = ['opponent_team']
        else:
            group_cols = ['team']

        group_keys = group_cols + ['season']
        df = df.sort_values(group_keys + ['week']).reset_index(drop = True)

        for m in metrics: 
            # Predictive season totals and averages
            prior_sum = df.groupby(group_keys)[m].cumsum().shift(1)
            prior_count = df.groupby(group_keys).cumcount()

            df[f'szn_total_{m}'] = prior_sum
            df[f'szn_avg_{m}'] = prior_sum / prior_count.replace(0, np.nan)
    df = df.copy()        
    return df

# Add rolling stats (pos)
def add_rolling_features(df, metrics, windows = [3,5], group = ['player_id', 'season']):
    df = df.copy()
    df = df.sort_values(group + ['week'])

    new_features = []

    for m in metrics:
        for w in windows:

            roll_avg = (
                df.groupby(group)[m]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

            roll_std = (
                df.groupby(group)[m]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).std())
            )

            roll_change = (
                df.groupby(group)[m]
                .transform(lambda s: (s.shift(1) - s.shift(2)) / s.shift(2).replace(0,np.nan))
            ).fillna(0)

            feat = pd.DataFrame({
                f'{m}_roll_avg_{w}': roll_avg,
                f'{m}_std_{w}': roll_std,
                f'{m}_roll_change_{w}': roll_change
            }, index = df.index)

            new_features.append(feat)

    df = pd.concat([df] + new_features, axis = 1)
    df = df.copy()
    return df

# Add rolling metrics (def) across the league
def add_league_rolling_averages(defense_df, metrics):
    df = defense_df.copy()
    df = df.sort_values(['season', 'week']).reset_index(drop = True)

    league = (
        df.groupby(['season', 'week'])[metrics]
          .mean()
          .reset_index()
          .sort_values(['season', 'week'])
    )

    def cum_prior_mean(g):
        return g.shift(1).expanding().mean()

    cum = league.groupby('season', group_keys = False)[metrics].apply(cum_prior_mean).reset_index(level = 0, drop = True)
    cum = pd.concat([league[['season', 'week']].reset_index(drop = True), cum.reset_index(drop = True)], axis = 1)

    cum = cum.rename(columns = {m: f'league_avg_{m}' for m in metrics})

    return cum, league

# ---------------------        -----      -----------------------------------------

# Create defensive features function
def build_defense_features(pbp, schedules, df):
    pbp = pbp.copy()
    schedules = schedules.copy()
    df = df.copy()

    # Map players
    player_map = df[['player_id', 'position']].drop_duplicates(subset = ['player_id'])

    pbp = pbp.merge(
        player_map,
        left_on = 'receiver_id',
        right_on = 'player_id',
        how = 'left'
    )

    pbp = pbp.merge(
        player_map,
        left_on = 'passer_id',
        right_on = 'player_id',
        how = 'left'
    )

    pbp = pbp.merge(
        player_map,
        left_on = 'rusher_id',
        right_on = 'player_id',
        how = 'left'
    )

    pbp['rush_yards_allowed'] = pbp['rushing_yards'].fillna(0)
    pbp['rec_yards_allowed'] = pbp['receiving_yards'].fillna(0)

    # Create yards allowed metrics
    pbp['pass_yards_allowed'] = pbp['passing_yards'].fillna(0)
    pbp['air_yards_allowed'] = pbp['air_yards'].fillna(0)
    pbp['rush_yards_allowed_qb'] = np.where(pbp['position'] == 'QB', pbp['rushing_yards'],0)
    pbp['rush_yards_allowed_rb'] = np.where(pbp['position'] == 'RB', pbp['rushing_yards'],0)

    pbp = pbp.copy()

    pbp['yards_after_catch_allowed'] = pbp['yards_after_catch'].fillna(0)
    pbp['rec_yards_allowed_wr'] = np.where(pbp['position'] == 'WR', pbp['receiving_yards'], 0)
    pbp['rec_yards_allowed_rb'] = np.where(pbp['position'] == 'RB', pbp['receiving_yards'], 0)
    pbp['rec_yards_allowed_te'] = np.where(pbp['position'] == 'TE', pbp['receiving_yards'], 0)   

    pbp = pbp.copy()

    # Create remaining metrics

    pbp['first_down_allowed'] = (pbp['first_down'] == 1).astype(int)
    pbp['completions_allowed'] = pbp['complete_pass'].fillna(0).astype(int)
    pbp['dropbacks_allowed'] = pbp['qb_dropback'].fillna(0).astype(int)

    # Carries
    pbp['carries_allowed'] = pbp['rush_attempt'].fillna(0)
    pbp['carries_allowed_rb'] = np.where((pbp['position'] == 'RB') & (pbp['rush_attempt'] == 1), 1, 0)
    pbp['carries_allowed_qb'] = np.where((pbp['position'] == 'QB') & (pbp['rush_attempt'] == 1), 1, 0)

    # Receptions
    pbp['receptions_allowed'] = pbp['complete_pass'].fillna(0)
    pbp['receptions_allowed_rb'] = np.where((pbp['position'] == 'RB') & (pbp['complete_pass'] == 1), 1, 0)
    pbp['receptions_allowed_wr'] = np.where((pbp['position'] == 'WR') & (pbp['complete_pass'] == 1), 1, 0)
    pbp['receptions_allowed_te'] = np.where((pbp['position'] == 'TE') & (pbp['complete_pass'] == 1), 1, 0)

    # Redzone metrics
    pbp['redzone_carries_allowed'] = np.where((pbp['is_redzone'] == 1) & (pbp['rush_attempt']), 1, 0)
    pbp['redzone_carries_allowed_rb'] = np.where((pbp['position'] == 'RB') &
                                                 (pbp['is_redzone'] == 1) & (pbp['rush_attempt']), 1, 0)
    pbp['redzone_carries_allowed_qb'] = np.where((pbp['position'] == 'QB') &
                                                 (pbp['is_redzone'] == 1) & (pbp['rush_attempt']), 1, 0)
    pbp['redzone_receptions_allowed'] = np.where((pbp['is_redzone'] == 1) & (pbp['complete_pass']), 1, 0)
    pbp['redzone_receptions_allowed_rb'] = np.where((pbp['position'] == 'RB') &
                                                 (pbp['is_redzone'] == 1) & (pbp['complete_pass']), 1, 0)
    pbp['redzone_receptions_allowed_wr'] = np.where((pbp['position'] == 'WR') &
                                                 (pbp['is_redzone'] == 1) & (pbp['complete_pass']), 1, 0)
    pbp['redzone_receptions_allowed_te'] = np.where((pbp['position'] == 'TE') &
                                                 (pbp['is_redzone'] == 1) & (pbp['complete_pass']), 1, 0)
    
    # Greenzone
    pbp['greenzone_carries_allowed'] = np.where((pbp['is_greenzone'] == 1) & (pbp['rush_attempt']), 1, 0)
    pbp['greenzone_carries_allowed_rb'] = np.where((pbp['position'] == 'RB') &
                                                 (pbp['is_greenzone'] == 1) & (pbp['rush_attempt']), 1, 0)
    pbp['greenzone_carries_allowed_qb'] = np.where((pbp['position'] == 'QB') &
                                                 (pbp['is_greenzone'] == 1) & (pbp['rush_attempt']), 1, 0)
    pbp['greenzone_receptions_allowed'] = np.where((pbp['is_greenzone'] == 1) & (pbp['complete_pass']), 1, 0)
    pbp['greenzone_receptions_allowed_rb'] = np.where((pbp['position'] == 'RB') &
                                                 (pbp['is_greenzone'] == 1) & (pbp['complete_pass']), 1, 0)
    pbp['greenzone_receptions_allowed_wr'] = np.where((pbp['position'] == 'WR') &
                                                 (pbp['is_greenzone'] == 1) & (pbp['complete_pass']), 1, 0)
    pbp['greenzone_receptions_allowed_te'] = np.where((pbp['position'] == 'TE') &
                                                 (pbp['is_greenzone'] == 1) & (pbp['complete_pass']), 1, 0)

    # Touchdowns
    pbp['pass_td_allowed'] = (pbp['pass_touchdown'] == 1).astype(int)
    pbp['rush_td_allowed_qb'] = np.where((pbp['position'] == 'QB') & (pbp['rush_touchdown'] == 1), 1, 0)
    pbp['rush_td_allowed_rb'] = np.where((pbp['position'] == 'RB') & (pbp['rush_touchdown'] == 1), 1, 0)
    pbp['rec_td_allowed_wr'] = np.where((pbp['position'] == 'WR') & (pbp['pass_touchdown'] == 1), 1, 0)
    pbp['rec_td_allowed_rb'] = np.where((pbp['position'] == 'RB') & (pbp['pass_touchdown'] == 1), 1, 0)
    pbp['rec_td_allowed_te'] = np.where((pbp['position'] == 'TE') & (pbp['pass_touchdown'] == 1), 1, 0)

    # Turnovers/Negative Plays
    pbp['interceptions_forced'] = pbp['interception'].fillna(0).astype(int)
    pbp['fumbles_forced'] = pbp['fumble_forced'].fillna(0).astype(int)
    pbp['sack_made'] = pbp['sack'].fillna(0).astype(int)
    pbp['negative_play_created'] = ((pbp['sack'] == 1) | (pbp['tackled_for_loss'] == 1)).astype(int)
    pbp['pressures'] = pbp['was_pressure'].fillna(0).astype(int)

    # EPA
    pbp['epa_allowed'] = pbp['epa'].fillna(0)
    pbp['pass_epa_allowed'] = pbp['air_epa'].fillna(0) + pbp['yac_epa']

    # Explosives

    pbp['explosive_pass_allowed'] = ((pbp['air_yards'].fillna(0) >= 20) | (pbp['yards_after_catch'].fillna(0) >= 20)).astype(int)
    pbp['explosive_rush_allowed'] = (pbp['rushing_yards'].fillna(0) >=10).astype(int)

    if 'defense_team' not in pbp.columns and 'defteam' in pbp.columns:
        pbp['defense_team'] = pbp['defteam']

    # Create defensive dataset
    pbp = pbp.copy()

    defense = (
        pbp.groupby(['defense_team', 'season','week'])
        .agg({

            # Passing
            'pass_yards_allowed': 'sum',
            'pass_td_allowed': 'sum',
            'sack_made': 'sum',
            'interceptions_forced': 'sum',
            'pressures': 'sum',
            'air_yards_allowed': 'sum',
            'yards_after_catch_allowed': 'sum',
            'completions_allowed': 'sum',
            'explosive_pass_allowed': 'sum',
            'pass_epa_allowed': 'sum',
            'dropbacks_allowed': 'sum',

            # Rushing
            'rush_yards_allowed': 'sum',
            'rush_yards_allowed_rb': 'sum',
            'rush_yards_allowed_qb': 'sum',
            'rush_td_allowed_rb': 'sum',
            'rush_td_allowed_qb': 'sum',
            'carries_allowed': 'sum',
            'carries_allowed_rb': 'sum',
            'carries_allowed_qb': 'sum',
            'fumbles_forced': 'sum',
            'explosive_rush_allowed': 'sum',

            # Receiving
            'receptions_allowed': 'sum',
            'receptions_allowed_rb': 'sum',
            'receptions_allowed_wr': 'sum',
            'receptions_allowed_te': 'sum',
            'rec_yards_allowed': 'sum',
            'rec_yards_allowed_wr': 'sum',
            'rec_yards_allowed_rb': 'sum',
            'rec_yards_allowed_te': 'sum',
            'rec_td_allowed_wr': 'sum',
            'rec_td_allowed_rb': 'sum',
            'rec_td_allowed_te': 'sum',

            # Other
            'first_down_allowed': 'sum',
            'epa_allowed': 'sum',
            'success': 'mean',
            'play_id': 'count',

            # Redzone/Greenzone
            'redzone_carries_allowed': 'sum',
            'redzone_carries_allowed_rb': 'sum',
            'redzone_carries_allowed_qb': 'sum',
            'greenzone_carries_allowed': 'sum',
            'greenzone_carries_allowed_rb': 'sum',
            'greenzone_carries_allowed_qb': 'sum',
            'redzone_receptions_allowed': 'sum',
            'redzone_receptions_allowed_rb': 'sum',
            'redzone_receptions_allowed_wr': 'sum',
            'redzone_receptions_allowed_te': 'sum',
            'greenzone_receptions_allowed': 'sum',
            'greenzone_receptions_allowed_rb': 'sum',
            'greenzone_receptions_allowed_wr': 'sum',
            'greenzone_receptions_allowed_te': 'sum'
        })
        .reset_index()
        .rename(columns = {'defense_team': 'opponent_team', 'play_id': 'plays'})
    )

    defense['total_yards_allowed'] = (
        defense['pass_yards_allowed'] +
        defense['rush_yards_allowed']
    )

    defense['epa_per_play'] = (
        defense['epa_allowed'] / defense['plays'].replace(0, np.nan)
    )

    defense['turnover_created'] = (
        defense['interceptions_forced'] + defense['fumbles_forced']
    )

    defense['rush_yards_per_attempt'] = (
        defense['rush_yards_allowed'] / defense['carries_allowed'].replace(0, np.nan)
    )

    defense['pass_yards_per_completion'] = (
        defense['pass_yards_allowed'] / ((defense['pass_yards_allowed'] > 0).astype(int)).replace(0, np.nan)
    )

    defense['turnover_rate'] = (
        defense['turnover_created'] / defense['plays'].replace(0, np.nan)
    )

    defense['sack_rate'] = (
        defense['sack_made'] / defense['plays'].replace(0, np.nan)
    )

    defense['pressure_rate'] = (
        defense['pressures'] / defense['dropbacks_allowed'].replace(0, np.nan)
    )

    defense['qb_fantasy_points_allowed'] = (
        (defense['pass_yards_allowed'] * 0.04) +
        (defense['pass_td_allowed'] * 4) +
        (defense['rush_yards_allowed_qb'] * 0.1) +
        (defense['rush_td_allowed_qb'] * 6)
    )

    defense['rb_fantasy_points_allowed'] = (
        (defense['rush_yards_allowed_rb'] * 0.1) +
        (defense['rush_td_allowed_rb'] * 6) +
        (defense['rec_yards_allowed_rb'] * 0.1) +
        (defense['completions_allowed'] * 1) +
        (defense['rec_td_allowed_rb'] * 6)
    )

    defense['wr_fantasy_points_allowed'] = (
        (defense['rec_yards_allowed_wr'] * 0.1) +
        (defense['completions_allowed'] * 1) +
        (defense['rec_td_allowed_wr'] * 6)
    )

    defense['te_fantasy_points_allowed'] = (
        (defense['rec_yards_allowed_te'] * 0.1) +
        (defense['completions_allowed'] * 1) +
        (defense['rec_td_allowed_te'] * 6)
    )

    defense = defense.copy()

    # Create defensive points allowed for home teams
    home_def = schedules[['season', 'week', 'home_team', 'away_score']].copy()
    home_def = home_def.rename(columns={
        'home_team': 'team',
        'away_score': 'points_allowed'
    })

    # Create defensive points allowed for away teams
    away_def = schedules[['season', 'week', 'away_team', 'home_score']].copy()
    away_def = away_def.rename(columns={
        'away_team': 'team',
        'home_score': 'points_allowed'
    })

    def_points = pd.concat([home_def, away_def], ignore_index = True)

    defense = defense.merge(def_points, left_on = ['season', 'week', 'opponent_team'], right_on = ['season', 'week', 'team'], how = 'left')

    defense = add_rolling_features(
        defense,
        metrics = OPPONENT_METRICS,
        windows = [3, 5],
        group = ['opponent_team', 'season']
    )

    defense = defense.copy()

    defense = add_season_metrics(defense, OPPONENT_METRICS, group_cols = ['opponent_team'])

    defense = defense.copy()

    return defense


# Create helper function to designate home/away flag 
def add_home_away_flags(pos_df, schedules):
    # Simplify schedules to unique team-week mapping
    home_flags = schedules[['season', 'week', 'home_team']].copy()
    home_flags['home_game'] = 1
    home_flags.rename(columns={'home_team': 'team'}, inplace=True)

    away_flags = schedules[['season', 'week', 'away_team']].copy()
    away_flags['home_game'] = 0
    away_flags.rename(columns={'away_team': 'team'}, inplace=True)

    # Combine into one clean mapping
    team_week_flags = pd.concat([home_flags, away_flags], ignore_index=True)

    # Merge safely on team identity
    merged = pos_df.merge(
        team_week_flags,
        left_on=['season', 'week', 'recent_team'],
        right_on=['season', 'week', 'team'],
        how='left'
    )

    merged.drop(columns=['team'], inplace=True, errors='ignore')

    # Ensure the column exists even for missing merges
    if 'home_game' not in merged.columns:
        merged['home_game'] = np.nan
    return merged

# Create helper function to create win/loss flags and win(loss) streak
def add_team_win_streaks(pos_df, schedules):
    sch = schedules.copy()
    if 'season_type' not in sch.columns and 'game_type' in sch.columns:
        # Rename to match used name
        sch = sch.rename(columns={'game_type':'season_type'})
    
    # Rename game types for later
    sch['season_type'] = sch['season_type'].replace({
        'REG': 'REG',
        'WC': 'POST',
        'DIV': 'POST',
        'CON': 'POST',
        'SB': 'POST'
    })

    # Grab what we need
    sch = sch[['season', 'season_type', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()

    # Create team results 
    home = sch[['season', 'season_type', 'week', 'home_team', 'home_score', 'away_score']].rename(columns={'home_team':'team'})
    home['team_win'] = (home['home_score'] > home['away_score']).astype(int)
    home = home[['season', 'season_type', 'week', 'team', 'team_win']]

    away = sch[['season', 'season_type', 'week', 'away_team', 'away_score', 'home_score']].rename(columns={'away_team':'team'})
    away['team_win'] = (away['away_score'] > away['home_score']).astype(int)
    away = away[['season', 'season_type', 'week', 'team', 'team_win']]

    team_results = pd.concat([home, away], ignore_index=True)

    # Order regular and post season properly for a season
    stype_order = {'REG': 1, 'POST': 2}
    team_results['stype_order'] = team_results['season_type'].map(stype_order).fillna(1)
    team_results = team_results.sort_values(['team', 'season', 'stype_order', 'week']).reset_index(drop=True)

    # Calculate streak
    def compute_streaks(x):
        # Streak counter 
        cnt = 0
        out = []
        for v in x:
            out.append(cnt)
            if v == 1:
                cnt += 1
            else:
                cnt = 0
        return pd.Series(out, index=x.index)
    
    team_results['team_win_streak'] = team_results.groupby(['team', 'season'], group_keys=False)['team_win'].apply(compute_streaks)

    # Merge streaks back into positional df
    pos_df = pos_df.merge(
        team_results[['season', 'week', 'team', 'team_win_streak']],
        left_on = ['season', 'week', 'recent_team'],
        right_on = ['season', 'week', 'team'],
        how = 'left',
        validate = 'm:1'
    )

    pos_df = pos_df.copy()

    # Add flag for outcome of prior game
    pos_df['team_won_last'] = (
        pos_df.groupby(['recent_team', 'season'])['team_win'].shift(1).fillna(0).astype(int)
    )

    pos_df.drop(columns=['team'], inplace=True, errors='ignore')
    return pos_df

# Create rolling 3 week features and general stats
def add_pos_stats(df, position):
    df = df.copy().sort_values(['player_id', 'season', 'week'])
    
    pos_map = {
        'QB': QB_METRICS['core'],
        'RB': RB_METRICS['core'],
        'WR': WR_METRICS['core'],
        'TE': TE_METRICS['core']
    }

    metrics_to_roll = pos_map[position]

    # Add rolling and season metrics 
    df = add_season_metrics(df, metrics_to_roll, group_cols = ['player_id'])
    df = add_rolling_features(df, metrics_to_roll, windows = [3,5], group = ['player_id', 'season'])
    df = df.copy()
    
    # Add position specific metrics
    if position == "QB":
        # Adjusted Yards Per Attempt (Passing Yards + 20 * Passing TD - 45 * Interceptions) / Attempts
        df['aypa'] = (
            df['passing_yards'] +
            (20 * df['passing_tds']) -
            (45 * df['interceptions'])
        ) / df['attempts'].replace(0, np.nan)
    
        # Completion percentage and efficiency metrics
        df['completion_pct'] = df['completions'] / df['attempts'].replace(0, np.nan)
        df['yards_per_attempt'] = df['passing_yards'] / df['attempts'].replace(0, np.nan)

        # QB Passing Yards per Team Snap (Normalize Pace of Play)
        df['passing_yards_per_snap'] = df['passing_yards'] / df['team_off_snaps'].replace(0, np.nan)

        # Momentum indicators
        df['prev_passing_yards'] = df.groupby(['player_id', 'season'])['passing_yards'].shift(1)
        df['prev_yards_minus_roll_3_mean'] = (
            df['prev_passing_yards'] - df['passing_yards_roll_avg_3']
        )

        df['passing_trend_3'] = (
            df.groupby(['player_id', 'season'])['passing_yards']
            .diff(periods=3)
        )

        df['completion_trend_3'] = (
            df.groupby(['player_id', 'season'])['completions']
            .transform(lambda s: (s - s.shift(1)) / s.shift(1))
            .reset_index(drop = True)
            .rolling(3).mean()
        )

        df['fp_per_db'] = df['fantasy_points'] / df['qb_dropback'].replace(0, np.nan)

        # Remaining rolling mean and standard deviation calculated
        for stat in ['completion_pct', 'yards_per_attempt', 'fp_per_db', 'aypa']:
            if stat in df.columns:
                df[f'{stat}_roll_3_mean'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).mean())
                df[f'{stat}_roll_3_std'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).std())

    if position == 'RB':
        df ['rush_yards_per_attempt'] = df['rushing_yards'] / df['carries'].replace(0, np.nan)

        # EPA Per Touch
        total_touches = df['carries'] + df['receptions'].fillna(0)
        total_epa = df['rushing_epa'].fillna(0) + df['receiving_epa'].fillna(0)
        df['epa_per_touch'] = total_epa / total_touches.replace(0, np.nan)

        # Usage
        df['carries_per_snap'] = df['carries'] / df['team_off_snaps'].replace(0, np.nan)

        # Rolling mean and standard deviation calculated
        for stat in ['rush_yards_per_attempt', 'epa_per_touch', 'carries_per_snap']:
            if stat in df.columns:
                df[f'{stat}_roll_3_mean'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).mean())
                df[f'{stat}_roll_3_std'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).std())

        df = df.copy()
    if position == 'WR':
        df ['rec_yards_per_target'] = df['receiving_yards'] / df['targets'].replace(0, np.nan)

        # Target Share Per Snap
        df['targets_per_snap'] = df['targets'] / df['team_off_snaps'].replace(0, np.nan)

        # Rolling mean and standard deviation calculated
        for stat in ['rec_yards_per_target', 'targets_per_snap']:
            if stat in df.columns:
                df[f'{stat}_roll_3_mean'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).mean())
                df[f'{stat}_roll_3_std'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).std())

    if position == 'TE':
        df ['rec_yards_per_target'] = df['receiving_yards'] / df['targets'].replace(0, np.nan)
        
        # Target Share
        df['targets_per_snap'] = df['targets'] / df['team_off_snaps'].replace(0, np.nan)

        # Rolling mean and standard deviation calculated
        for stat in ['rec_yards_per_target', 'targets_per_snap']:
            if stat in df.columns:
                df[f'{stat}_roll_3_mean'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).mean())
                df[f'{stat}_roll_3_std'] = df.groupby('player_id')[stat].transform(lambda x: x.shift(1).rolling(3).std())
    
    df = df.copy()
    return df

# Create defense merger
def merge_defense_features(pos_df, defense_df, position, mapping = POSITION_DEF_METRICS):
    pos_df = pos_df.copy()
    defense_df = defense_df.copy()
    
    metrics = mapping[position]

    metrics_exist =[m for m in metrics if m in defense_df.columns]
    if len(metrics_exist) == 0:
        raise ValueError(f'No metrics available for position {position} in defense_df')
    
    cum_league, _ = add_league_rolling_averages(defense_df, metrics_exist)

    for col in cum_league.columns:
        if col.startswith('league_avg_'):
            cum_league[col] = cum_league[col].fillna(cum_league[col].mean())
    
    merged_df = pos_df.merge(cum_league, on = ['season', 'week'], how = 'left')

    cols_to_keep = ['season', 'week', 'opponent_team']
    rename_map = {}

    selected_def_cols = []

    patterns = ['szn_avg_', 'szn_total_', '_roll_avg_', '_std_', '_roll_change_']

    # Fix prefixes for predictive features
    for col in defense_df.columns:
        if any(m in col for m in metrics_exist):
            if any(pat in col for pat in patterns):
                selected_def_cols.append(col)
                rename_map[col] = f'def_{col}'

                # Keep 3 week and make def_avg
                if '_roll_avg_3' in col:
                    base_metric = col.replace('_roll_avg_3', '')
                    defense_df[f'def_avg_{base_metric}'] = defense_df[col]
                    selected_def_cols.append(f'def_avg_{base_metric}')

    opp_stats = defense_df[cols_to_keep + selected_def_cols].copy()
    opp_stats = opp_stats.rename(columns = rename_map)

    final_df = merged_df.merge(
        opp_stats,
        on = ['season', 'week', 'opponent_team'],
        how = 'left'
    )
    # cum has league_avg_<metric_exist>
    return final_df

# Create final safe merge for all team context metrics
def merge_team_context(pos_df, schedules):
    # Win/loss flag creation
    sch = schedules.copy()
    sch['home_win'] = (sch['home_score'] > sch['away_score']).astype(int)
    sch['away_win'] = (sch['away_score'] > sch['home_score']).astype(int)

    # Team-level view for both home and away team
    home_games = sch[['season', 'week', 'home_team', 'away_team', 'home_win']].rename(
        columns={'home_team': 'team', 'away_team': 'opponent_team', 'home_win':'team_win'}
    )
    home_games['home_game'] = 1

    away_games = sch[['season', 'week', 'home_team', 'away_team', 'away_win']].rename(
        columns={'away_team': 'team', 'home_team': 'opponent_team', 'away_win':'team_win'}
    )
    away_games['home_game'] = 0

    # Combine into unified "team_games"
    team_games = pd.concat([home_games, away_games], ignore_index=True)
    
    team_games = team_games.dropna(subset=['team', 'opponent_team'])
    team_games.drop_duplicates(subset=['season', 'week', 'team'], inplace=True)

    # Merge
    merged = pos_df.merge(
        team_games,
        left_on = ['season', 'week', 'recent_team', 'opponent_team'],
        right_on = ['season', 'week', 'team', 'opponent_team'],
        how='left'
    )

    # Clean
    merged.drop(columns=['team'], inplace=True, errors='ignore')
    merged = merged.drop_duplicates(subset = ['player_id', 'season', 'week'])
    return merged

# Create defensive efficiency in respect to season
def add_efficiency_metrics(df, position):
    df = df.copy()

    if position == 'QB':
        off_map = QB_METRICS['core']
        def_map = QB_METRICS['defense']
    if position == 'RB':
        off_map = RB_METRICS['core']
        def_map = RB_METRICS['defense']
    if position == 'WR':
        off_map = WR_METRICS['core']
        def_map = WR_METRICS['defense']
    if position == 'TE':
        off_map = TE_METRICS['core']
        def_map = TE_METRICS['defense']
    
    for stat in off_map:
        roll_col = f'{stat}_roll_avg_5'
        league_col = f'league_avg_{stat}'

        if roll_col in df.columns and league_col in df.columns:
            df[f'player_{stat}_efficiency_index'] = (
                df[roll_col] / df[league_col].replace(0, np.nan)
            )

    for def_stat in def_map:
        d_roll_col = f'def_{def_stat}_roll_avg_5'
        d_league_col = f'league_avg_{def_stat}'

        if d_roll_col in df.columns and d_league_col in df.columns:
            df[f'def_{def_stat}_efficiency_index'] = (
                df[d_roll_col] / df[d_league_col].replace(0, np.nan)
            )

    df = df.copy()
    return df

# Build QB dataset (final)
def finalize_dataset(pos_df, schedules, defense, position):
    pos_df = pos_df.copy()
    schedules = schedules.copy()
    defense = defense.copy()

    pos_df = merge_team_context(pos_df, schedules)
    pos_df = add_team_win_streaks(pos_df, schedules)
    pos_df = add_efficiency_metrics(pos_df, position)

    # Bye week tracker
    pos_df['bye_last_week'] = (pos_df['week'] - pos_df.groupby(['player_id', 'season'])['week'].shift(1) > 1).astype(int)

    # season week index
    pos_df['season_week'] = pos_df['week']

    # Playoffs indicator
    pos_df['is_playoffs'] = (pos_df['season_type'] != 'REG').astype(int)

    pos_df = pos_df.copy()
    return pos_df