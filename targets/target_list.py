from utils.api_client import create_target_list_from_api

def get_exoplanet_targets(limit: int = 200, use_api: bool = True, cache: bool = True):
    """
    Get exoplanet targets, either from API or fallback list
    """
    if use_api:
        try:
            targets = create_target_list_from_api(limit=limit)
            if cache:
                from utils.api_client import save_targets_to_csv
                save_targets_to_csv(targets)
            return targets
        except Exception as e:
            print(f"[WARN] API failed, using fallback list: {e}")
    
    # Fallback manual list
    FALLBACK_TARGETS = [
        {
            'name': 'WASP-18 b',
            'tic_id': 'TYC 2117-914-1',
            'period_days': 0.94145299,
        },
        {
            'name': 'WASP-12 b',
            'tic_id': 'TYC 4340-1116-1', 
            'period_days': 1.091421,
        },
        {
            'name': 'WASP-19 b',
            'tic_id': 'TYC 6750-594-1',
            'period_days': 0.788839,
        },
        {
            'name': 'WASP-43 b',
            'tic_id': 'TYC 5533-112-1',
            'period_days': 0.813475,
        },
        {
            'name': 'WASP-121 b',
            'tic_id': 'TYC 7630-352-1',
            'period_days': 1.274925,
        },
        # Add more fallback targets as needed...
    ]
    
    return FALLBACK_TARGETS[:limit]

# Default export
EXOPLANET_TARGETS = get_exoplanet_targets(limit=50, use_api=False)  # Small fallback