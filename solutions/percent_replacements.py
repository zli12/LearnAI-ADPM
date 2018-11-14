df_counts = pd.DataFrame({'replacements' : df_maint.groupby(['comp']).count()['machineID'],
                          'failures' : df_fails.groupby(['failure']).count()['machineID']})

df_counts['percent_due_to_failure'] = df_counts['failures'] / df_counts['replacements']
df_counts