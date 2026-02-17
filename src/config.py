# Define suffix mapping for merges
MERGE_SUFFIXES = {
    'opp_def':'_opp',           # def stats
    'schedule': '_sched',       # schedule-based info
    'season_avg': '_season',    # league-season averages
    'team_results': '_res'      # team results & streak
}

# Define potential team name issues
TEAM_FIX = {'WSH':'WAS', 'JAC':'JAX'} # Add here if more discovered

QB_METRICS = {
    'main' : [
        'player_id', 'player_name', 'season', 'season_type', 'week', 'recent_team', 'opponent_team', 'position',
        'passing_yards', 'passing_tds', 'attempts', 'rushing_yards', 'rushing_tds', 'completions',
        'interceptions', 'sacks', 'fantasy_points', 'fantasy_points_ppr', 'carries', 'passing_air_yards',
        'passing_epa', 'rushing_epa', 'qb_dropback', 'epa_per_play', 'cpoe', 'redzone_attempt',
        'redzone_completion', 'greenzone_attempt', 'greenzone_completion', 'redzone_carries', 'greenzone_carries',
        'team_off_snaps', 'team_pass_attempts', 'neutral_pass_rate'
    ],
    'core': [
        'passing_yards', 'passing_tds', 'attempts', 'rushing_yards', 'rushing_tds', 'completions',
        'interceptions', 'sacks', 'fantasy_points', 'fantasy_points_ppr', 'carries',
        'passing_air_yards','passing_epa', 'rushing_epa', 'epa_per_play', 'cpoe', 'qb_dropback', 'redzone_attempt',
        'redzone_completion', 'greenzone_attempt', 'greenzone_completion', 'redzone_carries', 'greenzone_carries',
        'team_off_snaps', 'team_pass_attempts', 'neutral_pass_rate'
    ],
    'rolling': ['passing_yards', 'passing_tds', 'interceptions',
        'air_yards', 'cpoe', 'sack', 'epa',
        'pressure_rate', 'time_to_throw', 'agg_yards',
        'rush_yards', 'rush_tds', 'completions'],
    'efficiency': ['yards_per_attempt', 'td_rate', 'int_rate', 'epa'],
    'defense': [
        'pass_yards_allowed', 'pass_td_allowed', 'pass_epa_allowed', 'completions_allowed',
        'air_yards_allowed', 'yards_after_catch_allowed', 'pressure_rate', 'sack_rate', 'points_allowed',
        'turnover_rate', 'epa_per_play', 'explosive_pass_allowed', 'rush_yards_allowed_qb',
        'rush_td_allowed_qb', 'qb_fantasy_points_allowed', 'carries_allowed', 'carries_allowed_qb',
        'redzone_carries_allowed', 'greenzone_carries_allowed', 'redzone_receptions_allowed', 'greenzone_receptions_allowed',
        'redzone_carries_allowed_qb', 'greenzone_carries_allowed_qb'
    ]
}

RB_METRICS = {
    'main': [
        'player_id', 'player_name', 'season', 'season_type', 'week', 'recent_team', 'opponent_team', 'position',
        'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'carries', 'receptions',
        'target_share', 'targets', 'fantasy_points', 'fantasy_points_ppr', 'rushing_epa', 'success_rate', 'is_hvt',
        'redzone_carries', 'greenzone_carries', 'redzone_receptions', 'greenzone_receptions', 'receiving_epa',
        'team_off_snaps', 'team_pass_attempts', 'team_rush_attempts'
    ],
    'core': [
        'rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'carries', 'targets', 'receptions',
        'target_share', 'fantasy_points', 'fantasy_points_ppr', 'rushing_epa', 'success_rate', 'is_hvt',
        'redzone_carries', 'greenzone_carries', 'redzone_receptions', 'greenzone_receptions', 'receiving_epa',
        'team_off_snaps', 'team_pass_attempts'
    ],
    'efficiency': ['yards_per_carry', 'missed_tackle_rate', 'target_share'],
    'defense': [
        'rush_yards_allowed_rb', 'rush_td_allowed_rb', 'rush_yards_per_attempt', 'points_allowed'
        'explosive_rush_allowed', 'turnover_rate', 'success', 'rec_yards_allowed_rb', 'rec_td_allowed_rb',
        'rb_fantasy_points_allowed', 'carries_allowed', 'carries_allowed_rb', 'receptions_allowed', 'receptions_allowed_rb',
        'redzone_carries_allowed', 'redzone_carries_allowed_rb', 'greenzone_carries_allowed', 'greenzone_carries_allowed_rb',
        'redzone_receptions_allowed', 'redzone_receptions_allowed_rb', 'greenzone_receptions_allowed', 'greenzone_receptions_allowed_rb'
    ]
}

WR_METRICS = {
    'main': [
        'player_id', 'player_name', 'season', 'season_type', 'week', 'recent_team', 'opponent_team', 'position',
        'receiving_yards', 'receiving_tds', 'targets', 'receptions', 'receiving_air_yards',
        'target_share', 'fantasy_points', 'fantasy_points_ppr', 'receiving_epa', 'wopr', 'racr', 'pacr', 'is_hvt',
        'redzone_receptions', 'greenzone_receptions', 'team_pass_attempts', 'team_off_snaps'
    ],
    'core': [
        'receiving_yards', 'receiving_tds', 'targets', 'receptions', 'receiving_air_yards', 'is_hvt',
        'fantasy_points', 'fantasy_points_ppr', 'target_share', 'receiving_epa', 'wopr', 'racr', 'pacr',
        'redzone_receptions', 'greenzone_receptions', 'team_pass_attempts', 'team_off_snaps'
    ],
    'efficiency': ['yards_per_route', 'target_share'],
    'defense': [
        'rec_yards_allowed_wr', 'completions_allowed', 'air_yards_allowed', 'yards_after_catch_allowed',
        'explosive_pass_allowed', 'points_allowed', 'rec_td_allowed_wr', 'pass_epa_allowed', 'wr_fantasy_points_allowed',
        'receptions_allowed', 'receptions_allowed_wr', 'redzone_receptions_allowed', 'redzone_receptions_allowed_wr',
        'greenzone_receptions_allowed', 'greenzone_receptions_allowed_wr'
    ]
}

TE_METRICS = {
    'main': [
        'player_id', 'player_name', 'season', 'season_type', 'week', 'recent_team', 'opponent_team', 'position',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets', 'fantasy_points', 'fantasy_points_ppr',
        'receiving_air_yards','target_share', 'receiving_epa', 'wopr', 'is_hvt', 'racr',
        'redzone_receptions', 'greenzone_receptions', 'team_pass_attempts', 'team_off_snaps'
    ],
    'core': [
        'receiving_yards', 'receiving_tds', 'receptions', 'targets', 'fantasy_points', 'fantasy_points_ppr',
        'receiving_air_yards','target_share', 'receiving_epa', 'wopr', 'is_hvt', 'racr',
        'redzone_receptions', 'greenzone_receptions', 'team_pass_attempts', 'team_off_snaps'
    ],
    'efficiency': ['yards_per_route', 'target_share'],
    'defense': [
        'rec_yards_allowed_te', 'completions_allowed', 'air_yards_allowed', 'yards_after_catch_allowed',
        'explosive_pass_allowed', 'points_allowed', 'rec_td_allowed_te', 'pass_epa_allowed', 'te_fantasy_points_allowed',
        'redzone_receptions_allowed', 'redzone_receptions_allowed_te', 'greenzone_receptions_allowed', 'greenzone_receptions_allowed_te'
    ]
}

POSITION_DEF_METRICS = {
    'QB': QB_METRICS['defense'],
    'RB': RB_METRICS['defense'],
    'WR': WR_METRICS['defense'],
    'TE': TE_METRICS['defense']
}

OPPONENT_METRICS = [
    'pass_yards_allowed',
    'carries_allowed',
    'carries_allowed_rb',
    'carries_allowed_qb',
    'rush_yards_allowed_qb',
    'rush_yards_allowed_rb',
    'rush_td_allowed_qb',
    'rush_td_allowed_rb',
    'air_yards_allowed',
    'receptions_allowed',
    'receptions_allowed_rb',
    'receptions_allowed_wr',
    'receptions_allowed_te',
    'rec_yards_allowed_wr',
    'rec_yards_allowed_rb',
    'rec_yards_allowed_te',
    'rec_td_allowed_wr',
    'rec_td_allowed_rb',
    'rec_td_allowed_te',
    'yards_after_catch_allowed',
    'epa_allowed',
    'sack_made',
    'interceptions_forced',
    'turnover_created',
    'pressures',
    'explosive_pass_allowed',
    'fumbles_forced',
    'success',
    'epa_per_play',
    'qb_fantasy_points_allowed',
    'rb_fantasy_points_allowed',
    'wr_fantasy_points_allowed',
    'te_fantasy_points_allowed',
    'points_allowed'
]