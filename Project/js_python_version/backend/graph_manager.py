# -*- coding: utf-8 -*-
"""
Gestionnaire du graphe pour l'API
"""

import os
import pickle
import time
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
import pandas as pd

from generate_graph import search_naive, search_bfs, search_dijkstra, search_hybrid, compute_weighted_distance

class GraphManager:
    def __init__(self):
        self.graph = None
        self.ads_data = None
        self.graph_path = os.path.join(os.path.dirname(__file__), "advertising_graph.pkl")
        # Chemins des fichiers : par défaut les CSV existants
        self.nodes_file = 'adsSim_data_nodes.csv'
        self.ads_file = 'queries_structured.csv'
    
    def load_ads_data(self, ads_file: str = None):
        """
        Charge les données des ads depuis le CSV.
        """
        if ads_file is None:
            ads_file = self.ads_file
        
        self.ads_data = {}
        try:
            if not os.path.exists(ads_file):
                print(f"⚠️  Fichier ads introuvable: {ads_file}. Utilisez /upload-files pour uploader.")
                return
            
            ads_df = pd.read_csv(ads_file)
            if ads_df.empty:
                print(f"⚠️  Fichier ads vide: {ads_file}")
                return
            
            for _, row in ads_df.iterrows():
                ad_id = row['point_A']
                # CHANGÉ : Stocker comme liste (JSON serializable)
                Y_vector = [float(x) for x in row['Y_vector'].split(';')]
                D = row['D']
                self.ads_data[ad_id] = {'Y_vector': Y_vector, 'D': D}
            
            print(f"✅ Ads chargées : {len(self.ads_data)} ads depuis {ads_file}")
        
        except Exception as e:
            print(f"❌ Erreur chargement ads : {e}. ads_data reste vide.")
    
    def build_new_graph(self, k: int = 10):
        """
        Construit un nouveau graphe depuis les fichiers (priorité aux uploadés).
        """
        from generate_graph import build_graph, save_graph
        
        print(f"\n🔨 CONSTRUCTION D'UN NOUVEAU GRAPHE")
        print(f"  - Fichier nodes: {self.nodes_file}")
        print(f"  - Fichier ads: {self.ads_file}")
        print(f"  - K-NN k: {k}")
        
        start_time = time.time()
        
        # Charger les données depuis les fichiers (priorité aux uploadés)
        try:
            nodes_df = pd.read_csv(self.nodes_file)
            ads_df = pd.read_csv(self.ads_file)
        except FileNotFoundError as e:
            raise Exception(f"Fichier introuvable : {e}. Uploadez les fichiers via /upload-files ou vérifiez les noms.")
        
        # Construire le graphe
        self.graph = build_graph(nodes_df, k=k)
        
        # Sauvegarder
        save_graph(self.graph, os.path.dirname(self.graph_path))
        
        # Charger les ads
        self.load_ads_data(self.ads_file)
        
        build_time = time.time() - start_time
        
        print(f"✅ Graphe construit et sauvegardé en {build_time:.3f}s")
        
        stats = self._get_graph_stats()
        stats['build_time'] = build_time
        
        return stats
    
    def search_in_radius(self, node_id: str, ad_id: str, method: str = 'hybrid') -> List[str]:
        """
        Recherche depuis node_id en utilisant Y et D de ad_id.
        """
        if self.graph is None:
            raise Exception("Aucun graphe chargé")
        
        if node_id not in self.graph:
            raise Exception(f"Node {node_id} introuvable")
        
        if self.ads_data is None:
            self.load_ads_data()
        
        if ad_id not in self.ads_data:
            raise Exception(f"Ad {ad_id} introuvable")
        
        Y_vector = np.array(self.ads_data[ad_id]['Y_vector'])  
        radius_X = self.ads_data[ad_id]['D']
    
        if method == 'naive':
            nodes_found = search_naive(self.graph, node_id, Y_vector, radius_X)
        elif method == 'bfs':
            nodes_found = search_bfs(self.graph, node_id, Y_vector, radius_X)
        elif method == 'dijkstra':
            nodes_found = search_dijkstra(self.graph, node_id, Y_vector, radius_X)
        elif method == 'hybrid':
            nodes_found = search_hybrid(self.graph, node_id, Y_vector, radius_X)
        else:
            raise Exception(f"Méthode {method} inconnue")
        
        return [node_id for node_id, _ in nodes_found]
    
    def load_existing_graph(self) -> Dict:
        """Charge un graphe existant"""
        print(f" CHARGEMENT DU GRAPHE EXISTANT")
        
        if not os.path.exists(self.graph_path):
            print(f" Fichier introuvable: {self.graph_path}")
            raise FileNotFoundError(f"Aucun graphe sauvegardé trouvé à {self.graph_path}")
        
        start_time = time.time()
        
        print(f" Chargement depuis: {self.graph_path}")
        with open(self.graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        load_time = time.time() - start_time
        
        print(f" Graphe chargé en {load_time:.3f}s")
        
        if self.ads_data is None:
            self.load_ads_data()
        
        stats = self._get_graph_stats()
        stats['load_time'] = load_time
        
        return stats
    
    def _save_graph(self):
        """Sauvegarde le graphe"""
        if self.graph is None:
            raise Exception("Aucun graphe à sauvegarder")
        
        print(f" Sauvegarde vers: {self.graph_path}")
        
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        print(f" Graphe sauvegardé: {self.graph_path}")
    
    def _get_graph_stats(self) -> Dict:
        """Retourne les statistiques du graphe"""
        if self.graph is None:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'ad_nodes': 0,
                'regular_nodes': 0,
                'num_features': 0,
                'ads': []
            }
        
        regular_nodes = [n for n, d in self.graph.nodes(data=True) 
                        if d.get('node_type') == 'regular']
        ad_nodes = [n for n, d in self.graph.nodes(data=True) 
                   if d.get('node_type') == 'ad']
        
        num_features = 0
        for node_id, node_data in self.graph.nodes(data=True):
            if 'features' in node_data:
                num_features = len(node_data['features'])
                break
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'ad_nodes': len(ad_nodes),
            'regular_nodes': len(regular_nodes),
            'num_features': num_features,
            'ads': sorted(ad_nodes)
        }
    
    def get_graph_data(self, feature_indices: Tuple[int, int, int] = (0, 1, 2)) -> Dict:
        """
        Retourne les données du graphe pour la visualisation 3D
        """
        if self.graph is None:
            raise Exception("Aucun graphe chargé")
        
        print(f"\n Préparation des données 3D (features {feature_indices})...")
        
        fx, fy, fz = feature_indices
        nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            features = node_data.get('features')
            node_type = node_data.get('node_type', 'regular')
            
            if features is None:
                print(f"  Warning: Nœud {node_id} sans features, ignoré")
                continue
            
            node_dict = {
                'id': node_id,
                'type': node_type,
                'x': float(features[fx]),
                'y': float(features[fy]),
                'z': float(features[fz])
            }
            
            if node_type == 'ad':
                radius_D = node_data.get('radius_D')
                if radius_D is not None:
                    node_dict['radius_D'] = float(radius_D)
            
            nodes.append(node_dict)
        
        links = []
        for source, target, edge_data in self.graph.edges(data=True):
            links.append({
                'source': source,
                'target': target,
                'type': edge_data.get('edge_type', 'unknown'),
                'weight': float(edge_data.get('weight', 1.0))
            })
        
        print(f" Données préparées: {len(nodes)} nœuds, {len(links)} liens")
        
        return {
            'nodes': nodes,
            'links': links
        }

    def search_in_radius(self, ad_id: str, radius_X: float, method: str = 'hybrid') -> List[Tuple[str, float]]:
        """
        Recherche les nœuds dans le rayon X autour d'un ad
        
        Returns:
        - Liste de tuples (node_id, distance)
        """
        if self.graph is None:
            raise Exception("Aucun graphe chargé")
        
        if ad_id not in self.graph:
            raise Exception(f"Ad {ad_id} introuvable dans le graphe")
        
        ad_data = self.graph.nodes[ad_id]
        ad_features = ad_data['features']
        Y_vector = ad_data['Y_vector']
        
        print(f"\n🔍 Recherche autour de {ad_id}")
        print(f"   Rayon X: {radius_X:.4f}")
        print(f"   Méthode: {method}")
        
        nodes_found = []
        
        # Recherche selon la méthode
        if method == 'naive':
            nodes_found = self._search_naive(ad_features, Y_vector, radius_X)
        elif method == 'bfs':
            nodes_found = self._search_bfs(ad_id, ad_features, Y_vector, radius_X)
        elif method == 'dijkstra':
            nodes_found = self._search_dijkstra(ad_id, ad_features, Y_vector, radius_X)
        elif method == 'hybrid':
            # Choisir automatiquement la meilleure méthode
            radius_D = ad_data['radius_D']
            ratio = radius_X / radius_D
            
            if ratio <= 0.8:
                print(f"   → Dijkstra (X ≤ 0.8*D)")
                nodes_found = self._search_dijkstra(ad_id, ad_features, Y_vector, radius_X)
            elif ratio <= 1.5:
                print(f"   → BFS (0.8*D < X ≤ 1.5*D)")
                nodes_found = self._search_bfs(ad_id, ad_features, Y_vector, radius_X)
            else:
                print(f"   → Naive (X > 1.5*D)")
                nodes_found = self._search_naive(ad_features, Y_vector, radius_X)
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        print(f"✅ {len(nodes_found)} nœuds trouvés")
        
        return nodes_found

    def _search_naive(self, ad_features, Y_vector, radius_X) -> List[Tuple[str, float]]:
        """Recherche naïve (parcours complet)"""
        nodes_found = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') != 'regular':
                continue
            
            node_features = node_data['features']
            distance = compute_weighted_distance(ad_features, node_features, Y_vector)
            
            if distance <= radius_X:
                nodes_found.append((node_id, distance))  # CHANGÉ : Retourner tuple (id, distance)
        
        # Trier par distance croissante
        nodes_found.sort(key=lambda x: x[1])
        
        return nodes_found

    def _search_bfs(self, ad_id, ad_features, Y_vector, radius_X) -> List[Tuple[str, float]]:
        """Recherche BFS"""
        nodes_found = []
        visited = set()
        queue = [ad_id]
        visited.add(ad_id)
        
        while queue:
            current = queue.pop(0)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                
                if self.graph.nodes[neighbor].get('node_type') != 'regular':
                    continue
                
                neighbor_features = self.graph.nodes[neighbor]['features']
                distance = compute_weighted_distance(ad_features, neighbor_features, Y_vector)
                
                if distance <= radius_X:
                    nodes_found.append((neighbor, distance))  # CHANGÉ : Retourner tuple (id, distance)
                    queue.append(neighbor)
        
        # Trier par distance croissante
        nodes_found.sort(key=lambda x: x[1])
        
        return nodes_found

    def _search_dijkstra(self, ad_id, ad_features, Y_vector, radius_X) -> List[Tuple[str, float]]:
        """Recherche Dijkstra avec file de priorité"""
        import heapq
        
        nodes_found = []
        visited = set()
        heap = [(0, ad_id)]
        
        while heap:
            current_dist, current = heapq.heappop(heap)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in visited:
                    continue
                
                if self.graph.nodes[neighbor].get('node_type') != 'regular':
                    continue
                
                neighbor_features = self.graph.nodes[neighbor]['features']
                distance = compute_weighted_distance(ad_features, neighbor_features, Y_vector)
                
                if distance <= radius_X:
                    nodes_found.append((neighbor, distance))  # CHANGÉ : Retourner tuple (id, distance)
                    heapq.heappush(heap, (distance, neighbor))
        
        # Trier par distance croissante
        nodes_found.sort(key=lambda x: x[1])
        
        return nodes_found

# ...existing code...
# Singleton global
graph_manager = GraphManager()