# Crash_and_learn

<!-- Faire un circuit -->
docker compose run --rm recorder python framework/simulation/sim_recorder.py --controller submission --map Spa --laps 2

<!-- Lancer le replay interactif -->
docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix replay python framework/simulation/viz_replay.py --npz /app/recordings/episode_0003.npz --follow

Modifier ton environnement / entraînement
src/crashlearn/environment/environment.py
ou ton train.py selon ce que tu changes.

Supprimer l'ancien modèle

rm -f models/crashlearn_ppo.zip

Relancer l'entraînement

docker compose run --rm sim python train.py

Vérifier que le nouveau modèle a bien été créé

ls -lh models/crashlearn_ppo.zip

Réexporter le modèle en ONNX

docker compose run --rm recorder python export_onnx.py

Vérifier que les fichiers de submission sont présents

ls -lh submission/

Tu dois avoir :

agent.py
model.onnx

Tester le contrat de la submission

docker compose run --rm sim python test_contract.py

Valider la submission

docker compose run --rm sim python verify_submission.py --submission submission/

Enregistrer une nouvelle course

docker compose run --rm recorder python framework/simulation/sim_recorder.py \
  --controller submission \
  --map Spa \
  --laps 2

Regarder le résultat

Si tu veux directement une vidéo :

docker compose run --rm replay \
  python framework/simulation/viz_replay.py \
  --npz /app/recordings/episode_XXXX.npz \
  --record /app/recordings/test.mp4 \
  --no-telemetry

Ouvrir la vidéo

explorer.exe recordings/test.mp4

Si la configuration est bonne → conserver le modèle.

Si elle est mauvaise → revenir à l'étape 1, modifier la configuration et recommencer à 2.