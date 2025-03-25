-- Afficher les colonnes de la table logs
SELECT heure, pseudo, contexte, composant, evenement
FROM table_name
WHERE table_name = 'logs'
ORDER BY ordinal_position;

-- Afficher les colonnes de la table notes
SELECT pseudo, note
FROM table_name
WHERE table_name = 'notes'
ORDER BY ordinal_position;
