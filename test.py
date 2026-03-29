from backend.api import resolve_player_name, map_engine
import json
r = resolve_player_name('f0rsaken') or 'f0rsaken'
score = map_engine.get_player_map_score(r, 'Lotus')
with open('out.txt', 'w', encoding='utf-8') as f:
    f.write(f"Resolved: {repr(r)}\nScore: {json.dumps(score)}\n")
