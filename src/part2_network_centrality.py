'''
PART 2: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Build a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is inline with the standards we're using in this class 
'''

import numpy as np
import pandas as pd
import networkx as nx
import json
import os
from datetime import datetime

# Build the graph
g = nx.Graph()

# Set up your dataframe(s) -> the df that's output to a CSV should include at least the columns 'left_actor_name', '<->', 'right_actor_name'
with open("data/movies.json") as in_file:
    # Don't forget to comment your code
    for line in in_file:
        # Don't forget to include docstrings for all functions

        # Load the movie from this line
        this_movie = json.loads(line)
            
        # Create a node for every actor
        for actor_id,actor_name in tuple(this_movie['actors']):
        # add the actor to the graph    
        # Iterate through the list of actors, generating all pairs
        ## Starting with the first actor in the list, generate pairs with all subsequent actors
        ## then continue to second actor in the list and repeat
            g.add_node(actor_id, name=actor_name)

        
        i = 0 #counter
        for left_actor_id,left_actor_name in this_movie['actors']:
            for right_actor_id,right_actor_name in this_movie['actors'][i+1:]:

                # Get the current weight, if it exists
                current_wt = g[left_actor_id][right_actor_id]["weight"] if g.has_edge(left_actor_id, right_actor_id) else 0 
                
                # Add an edge for these actors
                if current_wt > 0:
                    g[left_actor_id][right_actor_id]["weight"] = current_wt + 1
                else:
                    g.add_edge(left_actor_id, right_actor_id, weight=1)
            i += 1



# Print the info below
print("Nodes:", len(g.nodes))

#Print the 10 the most central nodes
deg_cent = nx.degree_centrality(g)
top10 = sorted(deg_cent.items(), key=lambda kv: kv[1], reverse=True)[:10]
print("Top 10 by degree centrality:")
for actor_id, score in top10:
    print(f"{g.nodes[actor_id].get('name', actor_id)}: {score:.4f}")

# Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`
os.makedirs("data", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build the edge list dataframe with the three required columns
edge_rows = []
for u, v, data in g.edges(data=True):
    left_actor_name = g.nodes[u].get("name", str(u))
    right_actor_name = g.nodes[v].get("name", str(v))
    edge_rows.append({
        "left_actor_name": left_actor_name,
        "<->": "<->",
        "right_actor_name": right_actor_name
    })

out_path = os.path.join("data", f"network_centrality_{timestamp}.csv")
pd.DataFrame(edge_rows, columns=["left_actor_name", "<->", "right_actor_name"]).to_csv(out_path, index=False)
print(f"Wrote: {out_path}")