"""
Programa para Análise Fotométrica de Streaks de Satélites
Autor: Túlio R. Bittar
Data: 2025

Este programa permite análise fotométrica de streaks de satélite em imagens FITS,
incluindo seleção interativa de pontos, zoom interativo e análises estatísticas fundamentais.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import glob
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry
from photutils.background import Background2D
import pandas as pd
from scipy import stats
import warnings
from typing import List, Tuple
from detect_streak import StreakCoordinates, detect_streaks_in_fits
from detect_streak import detect_streaks_in_fits


warnings.filterwarnings('ignore')

class SatellitePhotometry:
    def __init__(self):
        self.current_image = None
        self.current_data = None
        self.current_header = None
        self.streak_points = []
        self.n_points = 10
        self.aperture_radius = 5.0
        self.annulus_inner = 8.0
        self.annulus_outer = 12.0
        self.results = []
        self.interface_elements = []

        # Variáveis para zoom
        self.zoom_active = False
        self.original_xlim = None
        self.original_ylim = None
        self.zoom_factor = 2.0
        self.zoom_patches = []  # Para armazenar círculos de zoom

        # Variáveis para confirmação visual
        self.streak_line = None
        self.analysis_points_plot = None
        self.aperture_circles = []

        # Armazena elementos vizuais
        self.point_annotations = []  # Para armazenar anotações de texto
        self.point_markers = []  # Para armazenar marcadores de pontos

        # Configuração da figura
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.subplots_adjust(bottom=0.2)

        # Some com o gráfico esquisitão do plot
        self.ax.set_visible(False)

        # Botões de controle
        self.setup_buttons()

    def setup_buttons(self):
        """Configura os botões da interface"""
        # Botão para carregar pasta
        ax_load = plt.axes([0.211, 0.042, 0.05, 0.112])
        self.btn_load = Button(ax_load, 'Carregar\nPasta')
        self.btn_load.on_clicked(self.load_folder)

        # Botão para zoom
        ax_zoom = plt.axes([0.2908, 0.1, 0.049, 0.04])
        self.btn_zoom = Button(ax_zoom, 'Zoom')
        self.btn_zoom.on_clicked(self.toggle_zoom)

        # Botão para resetar zoom
        ax_reset = plt.axes([0.2908, 0.05, 0.049, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset',
                                color = '#f0c559',
                                hovercolor = '#f4d37f')
        self.btn_reset.on_clicked(self.reset_zoom)


        # Botão para confirmar streak
        ax_confirm = plt.axes([0.344, 0.1, 0.12, 0.04])
        self.btn_confirm = Button(ax_confirm, 'Confirmar Streak')
        self.btn_confirm.on_clicked(self.confirm_streak)

        # Botão para resetar o streak
        ax_deny = plt.axes([0.468, 0.1, 0.08, 0.04])
        self.btn_deny = Button(ax_deny, 'Resetar Streak')
        self.btn_deny.on_clicked(self.reset_streak)

        # Botão para analisar
        ax_analyze = plt.axes([0.468, 0.05, 0.08, 0.04])
        self.btn_analyze = Button(ax_analyze, 'Analisar',
        color = 'lightgray',  # Cor inicial (desabilitado)
        hovercolor = 'gray')
        self.btn_analyze.on_clicked(
            lambda event: self.analyze_photometry(event, fast=True)
        )

        # Botão para próxima imagem
        ax_next = plt.axes([0.552, 0.05, 0.12, 0.04])
        self.btn_next = Button(ax_next, 'Próxima Imagem')
        self.btn_next.on_clicked(self.next_image)

        # Botão para a imagem anterior
        ax_prev = plt.axes([0.344, 0.05, 0.12, 0.04])
        self.btn_prev = Button(ax_prev, 'Imagem Anterior')
        self.btn_prev.on_clicked(self.previous_image)

        # Botão para visualizar pontos
        ax_show_points = plt.axes([0.552, 0.1, 0.12, 0.04])
        self.btn_show_points = Button(ax_show_points, 'Mostrar Pontos')
        self.btn_show_points.on_clicked(self.toggle_points_visibility)

        # Botão para identificar streak
        ax_find_streak = plt.axes([0.71, 0.05, 0.052, 0.092])
        self.btn_find_streak = Button(ax_find_streak, 'Identificar\nStreak ⚡',
                                      color='#71b4eb',
                                      hovercolor='#3ea4f7')

        self.btn_find_streak.on_clicked(self.find_streak)

        # Botão para realizar análise completa
        ax_complete_analsys = plt.axes([0.769, 0.05, 0.057, 0.092])
        self.btn_complete_analsys = Button(ax_complete_analsys, 'Análise >>\nCompleta',
                                           color='#fcbb7a',
                                           hovercolor='#eaa867')

        self.btn_complete_analsys.on_clicked(self.complete_analysis)

        # Botão de configurações
        ax_config = plt.axes([0.71, 0.881, 0.05, 0.04])
        self.btn_config = Button(ax_config, '⚙️ Config.',
                                      color='white',  # Cor inicial
                                      hovercolor='lightgray')
        self.btn_config.on_clicked(self.open_config_dialog)

        #ax_config.spines['top'].set_visible(False)
        #ax_config.spines['right'].set_visible(False)
        ax_config.spines['bottom'].set_visible(False)
        #ax_config.spines['left'].set_visible(False)


        # Detalhes da interface
        self.add_interface_graphics()

    def add_interface_graphics(self):

        # Caixa "selecione uma pasta"
        box_x = 0.267  # Posição X
        box_y = 0.209  # Posição Y
        box_width = 0.4931  # Largura da caixa
        box_height = 0.6702  # Altura da caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='black',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-2)  # especifica a ordem
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)

        # Texto selecione uma pasta
        hello_text = self.fig.text(0.53, 0.52, 'Para começar, clique em selecionar uma pasta!',
                                   ha='center', va='center',
                                   fontsize=13,
                                   fontfamily='Arial Rounded MT Bold',
                                   fontweight='normal',
                                   color='#4A4A4A',
                                   transform=self.fig.transFigure,
                                   zorder=-1)



        # Caixa do painel Semi
        box_x = 0.28  # Posição X (um pouco à esquerda do centro)
        box_y = 0.041  # Posição Y (abaixo do texto)
        box_width = 0.4  # Largura da caixa
        box_height = 0.11  # Altura da caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='black',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-2) # especifica a ordem
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)


        # Caixa do Texto "SEMI"
        box_x = 0.274  # Posição X (um pouco à esquerda do centro)
        box_y = 0.068  # Posição Y (abaixo do texto)
        box_width = 0.01  # Largura da caixa
        box_height = 0.06  # Altura da caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='white',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-1)
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)

        # Texto "SEMI"
        hello_text = self.fig.text(0.28, 0.098, 'SEMI',
                                   ha='center', va='center',
                                   fontsize=11,
                                   fontfamily='Arial Rounded MT Bold',
                                   fontweight='normal',
                                   color='black',
                                   rotation=90,
                                   transform=self.fig.transFigure,
                                   zorder=0)


        # CAIXA do painel Auto
        box_x = 0.70  # Posição X (um pouco à esquerda do centro)
        box_y = 0.041  # Posição Y (abaixo do texto)
        box_width = 0.135  # Largura da caixa
        box_height = 0.11  # Altura da caixa
        # Cria a caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='black',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-2)
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)

        # Caixa do Texto "AUTO"
        box_x = 0.694  # Posição X (um pouco à esquerda do centro)
        box_y = 0.068  # Posição Y (abaixo do texto)
        box_width = 0.01  # Largura da caixa
        box_height = 0.06  # Altura da caixa
        # Cria a caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='white',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-1)
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)

        # Texto "AUTO"
        hello_text = self.fig.text(0.70, 0.098, 'AUTO',
                                   ha='center', va='center',
                                   fontsize=11,
                                   fontfamily='Arial Rounded MT Bold',
                                   fontweight='normal',
                                   color='black',
                                   rotation=90,
                                   transform=self.fig.transFigure,
                                   zorder=0)


        # CAIXA do painel TIPS
        box_x = 0.78  # Posição X
        box_y = 0.21  # Posição Y
        box_width = 0.16  # Largura da caixa
        box_height = 0.14  # Altura da caixa
        # Cria a caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='black',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-2)
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)

        # Caixa do Texto "TIPS"
        box_x = 0.837  # Posição X
        box_y = 0.33  # Posição Y
        box_width = 0.0251  # Largura da caixa
        box_height = 0.04  # Altura da caixa
        # Cria a caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo
                                 edgecolor='white',  # Borda
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-1)
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)


        # Texto "TIPS"

        hello_text = self.fig.text(0.85, 0.35, 'TIPS',
                                   ha='center', va='center',
                                   fontsize=11,
                                   fontfamily='Arial Rounded MT Bold',
                                   fontweight='normal',
                                   color='black',
                                   transform=self.fig.transFigure,
                                   zorder=0)


        self.interface_elements.append(hello_text)

        # CAIXA do painel INFO
        box_x = 0.78  # Posição X (um pouco à esquerda do centro)
        box_y = 0.4  # Posição Y (abaixo do texto)
        box_width = 0.16  # Largura da caixa
        box_height = 0.2  # Altura da caixa
        # Cria a caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='black',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-2)
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)


        # Caixa do Texto "INFO"
        box_x = 0.836  # Posição X
        box_y = 0.58  # Posição Y
        box_width = 0.0274  # Largura da caixa
        box_height = 0.04  # Altura da caixa
        # Cria a caixa
        text_box = plt.Rectangle((box_x, box_y), box_width, box_height,
                                 facecolor='white',  # Fundo branco
                                 edgecolor='white',  # Borda preta
                                 linewidth=1,  # Espessura da borda
                                 transform=self.fig.transFigure,
                                 zorder=-1)
        self.fig.patches.append(text_box)
        self.interface_elements.append(text_box)

        # Texto "INFO"
        hello_text = self.fig.text(0.85, 0.6, 'INFO',
                                   ha='center', va='center',
                                   fontsize=11,
                                   fontfamily='Arial Rounded MT Bold',
                                   fontweight='normal',
                                   color='black',
                                   transform=self.fig.transFigure,
                                   zorder=0)


        self.interface_elements.append(hello_text)

    def load_folder(self, event=None):
        """Carrega pasta com imagens FITS"""
        root = tk.Tk()
        root.withdraw()


        # Altera a cor do botão


        folder_path = filedialog.askdirectory(title="Selecione a pasta com imagens FITS")
        if not folder_path:
            return

        # Busca arquivos FITS
        self.fits_files = glob.glob(os.path.join(folder_path, "*.fits"))
        self.fits_files.extend(glob.glob(os.path.join(folder_path, "*.fit")))

        if not self.fits_files:
            messagebox.showerror("Erro", "Nenhum arquivo FITS encontrado na pasta!")
            return

        self.current_file_index = 0

        # Solicita parâmetros
        self.get_parameters()

        # Carrega primeira imagem
        self.load_image()

    def get_parameters(self):
        """Solicita parâmetros do usuário"""
        root = tk.Tk()
        root.withdraw()

        # Configura a janela para aparecer na frente
        root.attributes('-topmost', True)
        root.update()

        # Número de pontos
        self.n_points = simpledialog.askinteger(
            "Parâmetros",
            "Número de pontos no streak:",
            initialvalue=13, minvalue=3, maxvalue=50, parent=root
        )

        # Raio da abertura
        self.aperture_radius = simpledialog.askfloat(
            "Parâmetros",
            "Raio da abertura (px):",
            initialvalue=4.0, minvalue=2.0, maxvalue=10.0, parent=root
        )

        # Anel de background
        self.annulus_inner = simpledialog.askfloat(
            "Parâmetros",
            "Raio interno do anel de background (px):",
            initialvalue=self.aperture_radius + 2, minvalue=self.aperture_radius + 1, parent=root
        )
        self.annulus_outer = simpledialog.askfloat(
            "Parâmetros",
            "Raio externo do anel de background (px):",
            initialvalue=self.aperture_radius + 6, minvalue=self.annulus_inner + 1, parent=root
        )

        # Zoom
        self.zoom_factor = simpledialog.askfloat(
            "Parâmetros",
            "Fator de zoom (e.g. 2.0 = 2×):",
            initialvalue=2.0, minvalue=1.1, maxvalue=10.0, parent=root
        )

        # Sigma-clipping
        self.sigma_clip_sigma = simpledialog.askfloat(
            "Parâmetros",
            "Sigma para sigma-clipped:",
            initialvalue=3.0, minvalue=1.0, maxvalue=5.0, parent=root
        )
        self.sigma_clip_maxiters = simpledialog.askinteger(
            "Parâmetros",
            "Iterações máximas de sigma-clipped:",
            initialvalue=5, minvalue=1, maxvalue=20, parent=root
        )

        # Saturação
        self.saturation_frac = simpledialog.askfloat(
            "Parâmetros",
            "Fraç. de saturação (0–1):",
            initialvalue=0.8, minvalue=0.5, maxvalue=1.0, parent=root
        )

        # Outliers
        self.zscore_thresh = simpledialog.askfloat(
            "Parâmetros",
            "Limiar Z‑score para outliers:",
            initialvalue=3.0, minvalue=1.0, maxvalue=5.0, parent=root
        )
        self.iqr_multiplier = simpledialog.askfloat(
            "Parâmetros",
            "Multiplicador IQR para outliers:",
            initialvalue=1.5, minvalue=0.5, maxvalue=3.0, parent=root
        )

        # Normalidade
        self.shapiro_pval_thresh = simpledialog.askfloat(
            "Parâmetros",
            "P‑valor mínimo no teste de normalidade:",
            initialvalue=0.05, minvalue=0.001, maxvalue=0.2, parent=root
        )

        root.destroy()

    def load_image(self):

        # Retoma o gráfico removido no início do programa
        self.ax.set_visible(True)

        """Carrega e exibe imagem FITS atual"""
        if not self.fits_files:
            return

        filename = self.fits_files[self.current_file_index]

        ######## Incluir no programa (em cima do gráfico)
        # print(f"Carregando: {os.path.basename(filename)}")

        try:
            with fits.open(filename) as hdul:
                self.current_data = hdul[0].data.astype(float)
                self.current_header = hdul[0].header

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar {filename}: {str(e)}")
            return

        # Limpa elementos anteriores
        self.clear_visual_elements()
        self.streak_points = []

        # Normalização da imagem para visualização
        norm = ImageNormalize(self.current_data, interval=ZScaleInterval())

        # Exibe imagem
        self.ax.clear()
        self.ax.imshow(self.current_data, cmap='gray', norm=norm, origin='lower')
        self.ax.set_title(f"Imagem: {os.path.basename(filename)}\nClique nos extremos do streak")
        # self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        # Salva limites originais
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        # Conecta eventos
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.draw()

    def clear_visual_elements(self):
        """Limpa elementos visuais anteriores"""
        # Remove linha do streak
        if self.streak_line:
            self.streak_line.remove()
            self.streak_line = None

        # Remove pontos de análise
        if self.analysis_points_plot:
            self.analysis_points_plot.remove()
            self.analysis_points_plot = None

        # Remove círculos de abertura
        for circle in self.aperture_circles:
            circle.remove()
        self.aperture_circles = []

        # Remove patches de zoom
        for patch in self.zoom_patches:
            patch.remove()
        self.zoom_patches = []

        # Remove anotações de texto
        for annotation in self.point_annotations:
            annotation.remove()
        self.point_annotations = []

        # Remove marcadores de pontos
        for marker in self.point_markers:
            marker.remove()
        self.point_markers = []

    def toggle_zoom(self, event=None):
        """Ativa/desativa modo zoom"""
        self.zoom_active = not self.zoom_active

        # Ativa a cor do botão de desativar o zoom
        #self.btn_reset.color = '#f0c559'
        #self.btn_reset.hovercolor = '#f4d37f'

        # Desativa a cor do botão de zoom
        #self.btn_zoom.color = 'lightgray'
        #self.btn_zoom.hovercolor = 'white'


        if self.zoom_active:
            self.ax.set_title(f"{self.ax.get_title()}\n MODO ZOOM ATIVO - Clique para fazer zoom")
            self.btn_zoom.label.set_text('Sair Zoom')
        else:
            title = self.ax.get_title().split('\n')[0]  # Remove linha do zoom
            self.ax.set_title(title)
            self.btn_zoom.label.set_text('Zoom')

        plt.draw()

    def reset_zoom(self, event=None):
        """Reseta zoom para visualização completa"""

        # Volta a cor do botão de reset ao normal
        #self.btn_reset.color = 'lightgray'
        #self.btn_reset.hovercolor = 'white'

        # Ativa a cor do botão de zoom
        #self.btn_zoom.color = '#f0c559'
        #self.btn_zoom.hovercolor = '#f4d37f'
        # self.clear_visual_elements()

        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            plt.draw()

    def reset_streak(self, event=None):
        """Reseta o streak e limpa todos os elementos visuais"""

        # Limpa todos os elementos visuais
        self.clear_visual_elements()

        # Retoma a cor do botão analisar e cancelar streak
        self.btn_analyze.color = 'lightgray'
        self.btn_analyze.hovercolor = 'gray'
        #self.btn_deny.color = 'lightgray'
        #self.btn_deny.hovercolor = 'gray'

        # Reseta variáveis do streak
        self.streak_points = []

        # Remove atributo analysis_points se existir
        if hasattr(self, 'analysis_points'):
            delattr(self, 'analysis_points')

        # Atualiza o título
        filename = os.path.basename(self.fits_files[self.current_file_index])
        self.ax.set_title(f"Imagem: {filename}\nClique nos extremos do streak")

        # Força redesenho
        plt.draw()

    def zoom_to_point(self, x, y):
        """Faz zoom em um ponto específico"""
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        # Calcula nova área de visualização
        x_range = (current_xlim[1] - current_xlim[0]) / self.zoom_factor
        y_range = (current_ylim[1] - current_ylim[0]) / self.zoom_factor

        new_xlim = [x - x_range / 3, x + x_range / 4]
        new_ylim = [y - y_range / 2, y + y_range / 2]

        # Garante que não saia dos limites da imagem
        img_height, img_width = self.current_data.shape
        new_xlim = [max(0, new_xlim[0]), min(img_width, new_xlim[1])]
        new_ylim = [max(0, new_ylim[0]), min(img_height, new_ylim[1])]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        # Adiciona círculo indicando área do zoom
        zoom_circle = plt.Circle((x, y), min(x_range, y_range) / 4,
                                 fill=False, color='yellow', linewidth=2, alpha=0.7)
        self.ax.add_patch(zoom_circle)
        self.zoom_patches.append(zoom_circle)

        plt.draw()

    def on_key_press(self, event):
        """Manipula teclas pressionadas"""
        if event.key == 'r':
            self.reset_zoom()
        elif event.key == 'z':
            self.toggle_zoom()
        elif event.key == 'p':
            self.toggle_points_visibility()

    def on_click(self, event):
        """Manipula cliques na imagem"""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        # Se modo zoom está ativo, faz zoom no ponto
        if self.zoom_active:
            self.zoom_to_point(x, y)
            return

        # Seleção de pontos do streak
        if len(self.streak_points) < 2:
            self.streak_points.append((x, y))

            # Marca ponto na imagem com visual melhorado
            point_marker = self.ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white',
                                        markeredgewidth=2, label=f'Ponto {len(self.streak_points)}')[0]

            # Armazena o marcador para remoção posterior
            self.point_markers.append(point_marker)

            # Adiciona texto identificando o ponto
            annotation = self.ax.annotate(f'P{len(self.streak_points)}', (x, y),
                                          xytext=(5, 5), textcoords='offset points',
                                          color='red', fontweight='bold', fontsize=12)

            # MODIFICADO: Armazena a anotação para remoção posterior
            self.point_annotations.append(annotation)

            if len(self.streak_points) == 1:
                self.ax.set_title(f"Ponto 1 selecionado ({x:.1f}, {y:.1f})\nClique no outro extremo do streak")
            else:
                # Desenha linha conectando os pontos
                x1, y1 = self.streak_points[0]
                x2, y2 = self.streak_points[1]
                self.streak_line = self.ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3,
                                                alpha=0.7, label='Streak')[0]

                self.ax.set_title(
                    f"Streak definido: P1({x1:.1f}, {y1:.1f}) → P2({x2:.1f}, {y2:.1f})\nClique em 'Confirmar Streak'")

            plt.draw()

    def confirm_streak(self, event=None):
        """Confirma e divide o streak em pontos"""
        if len(self.streak_points) != 2:
            messagebox.showwarning("Aviso", "Selecione dois pontos para definir o streak!")
            return

        # Desconecta evento de clique para seleção
        self.fig.canvas.mpl_disconnect(self.cid_click)

        # Calcula pontos ao longo do streak
        x1, y1 = self.streak_points[0]
        x2, y2 = self.streak_points[1]

        # Cria pontos equidistantes
        self.analysis_points = []
        for i in range(self.n_points):
            t = i / (self.n_points - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            self.analysis_points.append((x, y))

        # Visualiza pontos de análise com melhor visual
        # self.show_analysis_points()

        # Ativa o botão e o cancelar streak analisar visualmente
        self.btn_analyze.color = '#43de43'
        self.btn_analyze.hovercolor = '#90ee90'

        #self.btn_deny.color = '#f48f8f'
        #self.btn_deny.hovercolor = '#ecaeae'


        # Reconecta evento de clique (agora só para zoom)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.ax.set_title(f"{self.n_points} pontos de análise definidos\nClique em 'Analisar' para fotometria")
        plt.draw()

    def show_analysis_points(self):
        """Mostra pontos de análise com visualização detalhada"""
        if not hasattr(self, 'analysis_points'):
            return

        # Limpa visualizações anteriores dos pontos
        if self.analysis_points_plot:
            self.analysis_points_plot.remove()
        for circle in self.aperture_circles:
            circle.remove()
        self.aperture_circles = [0]

        # Plota pontos de análise
        xs, ys = zip(*self.analysis_points)
        self.analysis_points_plot = self.ax.plot(xs, ys, 'bo', markersize=8,
                                                 markeredgecolor='white', markeredgewidth=1,
                                                 label=f'{self.n_points} pontos')[0]

        # Desenha aberturas para cada ponto
        for i, (x, y) in enumerate(self.analysis_points):
            # Círculo da abertura fotométrica
            aperture_circle = plt.Circle((x, y), self.aperture_radius,
                                         fill=False, color='blue', linewidth=2, alpha=0.8)
            self.ax.add_patch(aperture_circle)
            self.aperture_circles.append(aperture_circle)

            # Anel de background (apenas borda externa para não poluir)
            background_circle = plt.Circle((x, y), self.annulus_outer,
                                           fill=False, color='red', linewidth=1,
                                           alpha=0.6, linestyle='--')
            self.ax.add_patch(background_circle)
            self.aperture_circles.append(background_circle)

            # Numeração dos pontos
            self.ax.annotate(f'{i + 1}', (x, y), xytext=(0, 0), textcoords='offset points',
                             ha='center', va='center', color='white', fontweight='bold',
                             fontsize=8, bbox=dict(boxstyle='circle,pad=0.1', facecolor='blue', alpha=0.7))

        # Atualiza legenda
        self.ax.legend(loc='upper right', fontsize=10)

    def toggle_points_visibility(self, event=None):
        """Alterna visibilidade dos pontos de análise"""
        if hasattr(self, 'analysis_points'):
            if self.analysis_points_plot and self.analysis_points_plot.get_visible():
                # Esconde elementos
                self.analysis_points_plot.set_visible(False)
                for circle in self.aperture_circles:
                    circle.set_visible(False)
                self.btn_show_points.label.set_text('Mostrar Pontos')
            else:
                # Mostra elementos
                if self.analysis_points_plot:
                    self.analysis_points_plot.set_visible(True)
                    for circle in self.aperture_circles:
                        circle.set_visible(True)
                else:
                    self.show_analysis_points()
                self.btn_show_points.label.set_text('Ocultar Pontos')

            plt.draw()

    def analyze_photometry(self, event=None, fast=None):
        """
        Realiza análise fotométrica completa de um streak de satélite.
        1) Verifica se todos os atributos necessários existem.
        1.1) Garante que a imagem do fundo fique
        2) Estima background global e local.
        3) Realiza fotometria em múltiplos pontos do streak.
        4) Calcula fluxo, ruído, S/N, magnitude e erros.
        5) Detecta outliers (Z-score e IQR) e realiza regressão linear.
        6) Plota resultados e salva dados.
        """
        # 1. Verificação de atributos essenciais
        required = [
            'analysis_points', 'aperture_radius', 'annulus_inner', 'annulus_outer',
            'current_data', 'current_header', 'ax'
        ]
        missing = [a for a in required if not hasattr(self, a)]
        if missing:
            messagebox.showerror("Erro", f"Faltando atributos: {', '.join(missing)}")
            return

        # 1.1
        self.show_analysis_points()

        # 2. Estima background global (2D) ou fallback sigma-clipped
        ny, nx = self.current_data.shape
        box = (max(10, min(50, ny // 10)), max(10, min(50, nx // 10)))
        try:
            bkg2d = Background2D(self.current_data, box, filter_size=(3, 3))
            global_bkg, global_bkg_std = bkg2d.background_median, bkg2d.background_rms_median
        except Exception:
            _, median, std = sigma_clipped_stats(self.current_data, sigma=3.0, maxiters=5)
            global_bkg, global_bkg_std = median, std

        # 3. Parâmetros do cabeçalho FITS
        hdr = self.current_header
        gain = hdr.get('GAIN', hdr.get('EGAIN', 1.0))
        readnoise = hdr.get('RDNOISE', hdr.get('READNOISE', 10.0))
        zp = hdr.get('ZEROPT', hdr.get('MAGZERO', 25.0))
        exptime = hdr.get('EXPTIME', 0.9)
        saturation_limit = hdr.get('SATURATE', 65535)

        # Verificação de parâmetros
        if exptime <= 0:
            messagebox.showerror("Erro", "Tempo de exposição inválido")
            return
        if gain <= 0:
            messagebox.showerror("Erro", "Ganho inválido")
            return

        # 4. Preparação de listas para resultados
        positions, fluxes, flux_errs = [], [], []
        snr_vals, bkg_vals = [], []
        mags, mag_errs = [], []
        saturated_flags = []

        # Métricas adicionais para análise científica
        fwhm_vals = []  # Para análise de qualidade
        centroid_shifts = []  # Para análise de movimento
        local_contrast = []  # Para análise de detectabilidade

        # Loop em cada ponto do streak
        for i, (x, y) in enumerate(self.analysis_points):
            positions.append(i)
            # Cria aperturas
            ap = CircularAperture((x, y), r=self.aperture_radius)
            an = CircularAnnulus((x, y), r_in=self.annulus_inner, r_out=self.annulus_outer)

            # Fotometria
            ft = aperture_photometry(self.current_data, ap)
            bt = aperture_photometry(self.current_data, an)

            # Background local médio
            bkg_mean = bt['aperture_sum'][0] / an.area
            bkg_vals.append(bkg_mean)

            # Flux bruto (ADU)
            raw_adu = ft['aperture_sum'][0] - bkg_mean * ap.area
            rate = raw_adu / exptime  # ADU/s

            # Verificação de saturação
            is_saturated = raw_adu > saturation_limit * 0.8
            saturated_flags.append(is_saturated)

            # Desvio local via sigma-clipped
            try:
                cut = an.to_mask(method='exact').cutout(self.current_data)
                _, _, std_local = sigma_clipped_stats(cut, sigma=3.0, maxiters=5)
            except Exception:
                std_local = global_bkg_std

            # Cálculo de ruído
            # Variância em elétrons: fonte + background + leitura
            var_source_e = max(0, raw_adu * gain)  # ruído de Poisson do sinal (elétrons)
            var_bkg_e = ap.area * (std_local ** 2) * gain  # ruído de fundo (elétrons)
            var_read_e = ap.area * readnoise ** 2  # ruído de leitura (elétrons)

            total_var_e = var_source_e + var_bkg_e + var_read_e
            noise_e = np.sqrt(total_var_e)
            noise_adu = noise_e / gain  # converter de volta para ADU

            # Erro no fluxo (ADU/s)
            err_rate = noise_adu / exptime

            # Relação S/N
            snr = rate / err_rate if err_rate > 0 else 0

            # Magnitude com verificação
            if rate > 0 and not is_saturated:
                # Para streak (movimento), a magnitude é instantânea
                mag = -2.5 * np.log10(rate) + zp
                mag_err = (2.5 / np.log(10)) * (err_rate / rate) if rate > 0 else np.nan
            else:
                mag, mag_err = np.nan, np.nan

            # Métricas adicionais para cada ponto
            # FWHM local (estimativa simples)
            try:
                cutout = ap.to_mask(method='exact').cutout(self.current_data)
                if cutout is not None:
                    # Estimativa de FWHM via momentos
                    fwhm_est = self.estimate_fwhm_from_cutout(cutout)
                    fwhm_vals.append(fwhm_est)
                else:
                    fwhm_vals.append(np.nan)
            except:
                fwhm_vals.append(np.nan)

            # Contraste local (sinal/background)
            local_contrast.append(rate / bkg_mean if bkg_mean > 0 else np.nan)

            # Armazena valores principais
            fluxes.append(rate)
            flux_errs.append(err_rate)
            snr_vals.append(snr)
            mags.append(mag)
            mag_errs.append(mag_err)

        # Converte a arrays
        fluxes, flux_errs = np.array(fluxes), np.array(flux_errs)
        snr_vals, bkg_vals = np.array(snr_vals), np.array(bkg_vals)
        mags, mag_errs = np.array(mags), np.array(mag_errs)
        positions = np.array(positions)
        saturated_flags = np.array(saturated_flags)
        fwhm_vals = np.array(fwhm_vals)
        local_contrast = np.array(local_contrast)

        # 5. Filtra pontos válidos (não saturados e com fluxo positivo)
        valid = (fluxes > 0) & ~np.isnan(mags) & ~saturated_flags
        if np.sum(valid) / len(fluxes) < 0.5:
            messagebox.showwarning("Aviso", f"Menos de 50% dos pontos válidos. Saturados: {np.sum(saturated_flags)}")

        # 6. Teste Shapiro-Wilk
        if 3 <= np.sum(valid) < 5000:
            sh_stat, sh_p = stats.shapiro(fluxes[valid])
        else:
            sh_stat, sh_p = np.nan, np.nan

        # 7. Outliers: Z-score e IQR
        if np.sum(valid) >= 2:
            z = np.abs(stats.zscore(fluxes[valid]))
            out_z = np.where(z > 3)[0]
            q1, q3 = np.percentile(fluxes[valid], [25, 75])
            iqr = q3 - q1
            out_iqr = np.where((fluxes[valid] < q1 - 1.5 * iqr) | (fluxes[valid] > q3 + 1.5 * iqr))[0]
        else:
            out_z, out_iqr = np.array([]), np.array([])

        # 8. Regressão linear
        if np.sum(valid) >= 2:
            slope, inter, r_val, p_val, _ = stats.linregress(positions[valid], fluxes[valid])
        else:
            slope, inter, r_val = np.nan, np.nan, np.nan

        # NOVAS MÉTRICAS CIENTÍFICAS (depois de definir valid e slope)
        # 1. Análise de movimento e geometria do streak
        trail_length = self.calculate_trail_length(self.analysis_points)
        angular_velocity = self.calculate_angular_velocity(self.analysis_points, exptime)
        position_angle = self.calculate_position_angle(self.analysis_points)
        streak_straightness = self.calculate_streak_straightness(self.analysis_points)

        # 2. Análise de variabilidade
        if np.sum(valid) >= 3:
            # Coeficiente de variação
            cv = np.nanstd(fluxes[valid]) / np.nanmean(fluxes[valid])

            # Análise de periodicidade (importante para rotação de satélites)
            from scipy.signal import periodogram
            try:
                freqs, power = periodogram(fluxes[valid] - np.nanmean(fluxes[valid]))
                dominant_freq = freqs[np.argmax(power[1:])] if len(power) > 1 else np.nan
                dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.nan
            except:
                dominant_period = np.nan

            # Teste de normalidade dos resíduos
            if not np.isnan(slope) and np.sum(valid) >= 3:
                residuals = fluxes[valid] - (inter + slope * positions[valid])
                _, normality_p = stats.shapiro(residuals)
            else:
                normality_p = np.nan
        else:
            cv = np.nan
            dominant_period = np.nan
            normality_p = np.nan

        # 3. Métricas de qualidade observacional
        mean_fwhm = np.nanmean(fwhm_vals[valid]) if np.sum(valid) > 0 else np.nan
        mean_contrast = np.nanmean(local_contrast[valid]) if np.sum(valid) > 0 else np.nan

        # 4. Parâmetros atmosféricos/instrumentais
        airmass = hdr.get('AIRMASS', np.nan)
        seeing = hdr.get('SEEING', np.nan)
        filter_name = hdr.get('FILTER', 'Unknown')

        # 5. Estatísticas robustas
        if np.sum(valid) >= 3:
            median_flux = np.nanmedian(fluxes[valid])
            mad_flux = np.nanmedian(np.abs(fluxes[valid] - median_flux))  # Median Absolute Deviation
            robust_std = 1.4826 * mad_flux  # Conversão para desvio padrão equivalente
        else:
            median_flux = np.nan
            mad_flux = np.nan
            robust_std = np.nan


        # 9. Plot com informações
        #self.ax.clear() Jeito antigo de fazer
        for line in self.ax.lines:
            if line.get_label() and 'Ajuste' in line.get_label():
                line.remove()

        for line in self.ax.lines:
            if line.get_label() and 'Ajuste' in line.get_label():
                line.remove()

        # Plot pontos válidos
        mask_valid = valid & ~saturated_flags
        if np.sum(mask_valid) > 0:
            self.ax.errorbar(positions[mask_valid], fluxes[mask_valid],
                             yerr=flux_errs[mask_valid], fmt='o', label='Fluxo válido')

        # Plot pontos saturados
        if np.sum(saturated_flags) > 0:
            self.ax.errorbar(positions[saturated_flags], fluxes[saturated_flags],
                             yerr=flux_errs[saturated_flags], fmt='s', color='red',
                             label='Saturado')

        # Linha de ajuste
        if not np.isnan(slope):
            self.ax.plot(positions, inter + slope * positions,
                         label=f'Ajuste (R²={r_val ** 2:.3f})')

        # Estatísticas no título
        valid_flux_mean = np.nanmean(fluxes[valid]) if np.sum(valid) > 0 else np.nan
        valid_snr_mean = np.nanmean(snr_vals[valid]) if np.sum(valid) > 0 else np.nan

        self.ax.set(
            title=f"Fluxo={valid_flux_mean:.1f} ADU/s, SNR={valid_snr_mean:.1f}, "
                  f"Válidos={np.sum(valid)}/{len(fluxes)}",
            xlabel='Ponto', ylabel='Fluxo (ADU/s)'
        )
        self.ax.legend()
        plt.draw()

        # 10. Salvar resultados expandidos para análise científica
        self.save_scientific_results({
            'basic_photometry': {
                'fluxes': fluxes, 'flux_errs': flux_errs, 'mags': mags, 'mag_errs': mag_errs,
                'snr_vals': snr_vals, 'bkg_vals': bkg_vals, 'positions': positions
            },
            'quality_metrics': {
                'fwhm_vals': fwhm_vals, 'local_contrast': local_contrast,
                'saturated_flags': saturated_flags, 'valid_flags': valid
            },
            'variability_analysis': {
                'coefficient_variation': cv, 'dominant_period': dominant_period,
                'normality_p_value': normality_p, 'median_flux': median_flux,
                'mad_flux': mad_flux, 'robust_std': robust_std
            },
            'geometric_analysis': {
                'trail_length': trail_length, 'angular_velocity': angular_velocity,
                'position_angle': position_angle, 'streak_straightness': streak_straightness
            },
            'observational_conditions': {
                'airmass': airmass, 'seeing': seeing, 'filter_name': filter_name,
                'exptime': exptime, 'mean_fwhm': mean_fwhm, 'mean_contrast': mean_contrast
            },
            'statistical_analysis': {
                'linear_fit': {'slope': slope, 'intercept': inter, 'r_squared': r_val ** 2},
                'outliers': {'z_score': out_z, 'iqr': out_iqr},
                'shapiro_wilk': {'statistic': sh_stat, 'p_value': sh_p}
            }
        })

        # Plot expandido e relatório final
        self.plot_results(positions, fluxes, flux_errs, snr_vals, mags, mag_errs,
                                        fwhm_vals, local_contrast, out_z, out_iqr, saturated_flags,fast)

        # Relatório de qualidade
        quality_report = self.generate_quality_report(valid, saturated_flags, cv, mean_fwhm,
                                                      mean_contrast, dominant_period)


        #messagebox.showinfo("Análise Concluída",
        #                    f"Fotometria científica concluída!\n"
        #                   f"Pontos válidos: {np.sum(valid)}/{len(fluxes)}\n"
        #                    f"Coef. variação: {cv:.3f}\n"
        #                    f"FWHM médio: {mean_fwhm:.2f} px\n"
        #                    f"Contraste médio: {mean_contrast:.1f}\n"
        #                    f"{quality_report}")

    def estimate_fwhm_from_cutout(self, cutout):
        """Estima FWHM de um cutout usando momentos de segunda ordem"""
        if cutout is None or np.all(cutout == 0):
            return np.nan
        try:
            # Método simples usando
            y, x = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
            total = np.sum(cutout)
            if total <= 0:
                return np.nan

            # Momentos
            x_mean = np.sum(x * cutout) / total
            y_mean = np.sum(y * cutout) / total
            x_var = np.sum((x - x_mean) ** 2 * cutout) / total
            y_var = np.sum((y - y_mean) ** 2 * cutout) / total

            # FWHM = 2.355 * sigma (para gaussiana)
            fwhm = 2.355 * np.sqrt((x_var + y_var) / 2)
            return fwhm
        except:
            return np.nan

    def calculate_trail_length(self, analysis_points):
        """Calcula o comprimento total do trail em pixels"""
        if len(analysis_points) < 2:
            return np.nan
        points = np.array(analysis_points)
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        return np.sum(distances)

    def calculate_angular_velocity(self,analysis_points, exptime):
        """Calcula velocidade angular em pixels/segundo"""
        trail_length = self.calculate_trail_length(analysis_points)
        return trail_length / exptime if exptime > 0 else np.nan

    def calculate_position_angle(self, analysis_points):
        """Calcula ângulo de posição do streak em graus"""
        if len(analysis_points) < 2:
            return np.nan
        points = np.array(analysis_points)
        dx = points[-1, 0] - points[0, 0]
        dy = points[-1, 1] - points[0, 1]
        return np.degrees(np.arctan2(dy, dx))

    def calculate_streak_straightness(self, analysis_points):
        """Calcula quão reto é o streak (R² do ajuste linear)"""
        # É legal ter para métricas posteriores
        if len(analysis_points) < 3:
            return np.nan
        points = np.array(analysis_points)
        # Ajuste linear para x vs posição
        t = np.arange(len(points))
        try:
            _, _, r_x, _, _ = stats.linregress(t, points[:, 0])
            _, _, r_y, _, _ = stats.linregress(t, points[:, 1])
            return (r_x ** 2 + r_y ** 2) / 2  # R² médio
        except:
            return np.nan

    def save_results(self, positions, fluxes, flux_errors, backgrounds,
                     snr_values, magnitudes, mag_errors):
        """Salva resultados em arquivo CSV"""

        # Cria DataFrame com resultados
        data = {
            'filename': [os.path.basename(self.fits_files[self.current_file_index])] * len(positions),
            'point': range(1, len(positions) + 1),
            'x': [pos[0] for pos in positions],
            'y': [pos[1] for pos in positions],
            'flux': fluxes,
            'flux_error': flux_errors,
            'background': backgrounds,
            'snr': snr_values,
            'magnitude': magnitudes,
            'mag_error': mag_errors
        }

        df = pd.DataFrame(data)

        # Nome do arquivo de saída
        base_name = os.path.splitext(os.path.basename(self.fits_files[self.current_file_index]))[0]
        output_file = f"{base_name}_photometry.csv"

        # Salva
        df.to_csv(output_file, index=False)
        print(f"\nResultados salvos em: {output_file}")

        # Adiciona aos resultados globais
        self.results.append(df)

    def next_image(self, event=None):
        """Carrega próxima imagem"""
        if not self.fits_files:
            return

        self.current_file_index += 1

        if self.current_file_index >= len(self.fits_files):
            messagebox.showinfo("Fim", "Todas as imagens foram processadas!")

            # Salva resultados obtidos
            if self.results:
                consolidated = pd.concat(self.results, ignore_index=True)
                consolidated.to_csv("consolidated_photometry.csv", index=False)
                print("\nResultados consolidados salvos em: consolidated_photometry.csv")

            return

        # Retoma a cor do botão analisar
        self.btn_analyze.color = 'lightgray'
        self.btn_analyze.hovercolor = 'gray'


        # Retoma a cor do botão de cancelar
        #self.btn_deny.color = '#f48f8f'
        #self.btn_deny.hovercolor = '#ecaeae'

        # Carrega próxima imagem
        self.load_image()


    def previous_image(self, event=None):
        """Carrega a imagem anterior"""
        if not self.fits_files:
            return

        self.current_file_index -= 1

        if self.current_file_index <= -1:
            messagebox.showinfo("Início", "Primeira imagem da pasta!")

            # Salva resultados obtidos
            if self.results:
                consolidated = pd.concat(self.results, ignore_index=True)
                consolidated.to_csv("consolidated_photometry.csv", index=False)
                print("\nResultados consolidados salvos em: consolidated_photometry.csv")

            return

        # Retoma a cor do botão analisar
        self.btn_analyze.color = 'lightgray'
        self.btn_analyze.hovercolor = 'gray'

        # Carrega próxima imagem
        self.load_image()

    def run(self):
        """Inicia o programa"""

        # Essas instruções serão  adicionadas mais tarde em "TIPs"
        # No momento é um placeholder
        print("=" * 60)
        print("ANÁLISE FOTOMÉTRICA DE STREAKS DE SATÉLITE")
        print("=" * 60)
        print("Instruções do modo semi-automático:")
        print("1. Clique em 'Carregar Pasta' para selecionar as imagens FITS")
        print("2. Clique nos dois extremos do streak do satélite")
        print("3. Clique em 'Confirmar Streak' para dividir em pontos")
        print("4. Clique em 'Analisar' para executar a fotometria")
        print("5. Clique em 'Próxima Imagem' para processar a próxima")
        print("\nFuncionalidades extras:")
        print("- 'Zoom': Ativa modo zoom - clique na imagem para fazer zoom")
        print("- 'Reset': Reseta zoom para visão completa")
        print("- 'Mostrar/Ocultar Pontos': Alterna visibilidade dos pontos")
        print("- Teclas de atalho: 'z' (zoom), 'r' (reset), 'p' (pontos)")
        print("=" * 60)

        plt.show()

    def open_config_dialog(self, event=None):
        """Abre janela de configurações para editar parâmetros"""

        # Cria janela principal
        config_window = tk.Toplevel()
        config_window.title("Configurações de Análise")
        config_window.geometry("400x350")
        config_window.resizable(False, False)

        # Centraliza a janela
        config_window.transient()
        config_window.grab_set()

        # Mantém a janela sempre no topo
        config_window.attributes('-topmost', True)

        # Variáveis para armazenar os valores
        n_points_var = tk.IntVar(value=self.n_points)
        aperture_var = tk.DoubleVar(value=self.aperture_radius)
        inner_var = tk.DoubleVar(value=self.annulus_inner)
        outer_var = tk.DoubleVar(value=self.annulus_outer)
        zoom_var = tk.DoubleVar(value=self.zoom_factor)

        # Título da janela
        title_label = tk.Label(config_window, text="Configurações",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        # Frame principal
        main_frame = tk.Frame(config_window)
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)

        # Função para validar entrada numérica
        def validate_number(value, var_type):
            if value == "":
                return True
            try:
                if var_type == int:
                    int(value)
                else:
                    float(value)
                return True
            except ValueError:
                return False

        # Registra função de validação
        vcmd_int = (config_window.register(lambda v: validate_number(v, int)), '%P')
        vcmd_float = (config_window.register(lambda v: validate_number(v, float)), '%P')

        # 1. Número de pontos
        tk.Label(main_frame, text="Número de pontos no streak:",
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))

        points_frame = tk.Frame(main_frame)
        points_frame.pack(fill='x', pady=(0, 15))

        points_entry = tk.Entry(points_frame, textvariable=n_points_var,
                                validate='key', validatecommand=vcmd_int,
                                font=('Arial', 10), width=10)
        points_entry.pack(side='left')

        tk.Label(points_frame, text="(mínimo: 3, máximo: 50)",
                 font=('Arial', 9), fg='gray').pack(side='left', padx=(10, 0))

        # 2. Raio da abertura
        tk.Label(main_frame, text="Raio da abertura (pixels):",
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))

        aperture_frame = tk.Frame(main_frame)
        aperture_frame.pack(fill='x', pady=(0, 15))

        aperture_entry = tk.Entry(aperture_frame, textvariable=aperture_var,
                                  validate='key', validatecommand=vcmd_float,
                                  font=('Arial', 10), width=10)
        aperture_entry.pack(side='left')

        tk.Label(aperture_frame, text="(mínimo: 1.0, máximo: 20.0)",
                 font=('Arial', 9), fg='gray').pack(side='left', padx=(10, 0))

        # 3. Raio interno do anel
        tk.Label(main_frame, text="Raio interno do anel de background:",
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))

        inner_frame = tk.Frame(main_frame)
        inner_frame.pack(fill='x', pady=(0, 15))

        inner_entry = tk.Entry(inner_frame, textvariable=inner_var,
                               validate='key', validatecommand=vcmd_float,
                               font=('Arial', 10), width=10)
        inner_entry.pack(side='left')

        tk.Label(inner_frame, text="(deve ser > raio da abertura)",
                 font=('Arial', 9), fg='gray').pack(side='left', padx=(10, 0))

        # 4. Raio externo do anel
        tk.Label(main_frame, text="Raio externo do anel de background:",
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))

        outer_frame = tk.Frame(main_frame)
        outer_frame.pack(fill='x', pady=(0, 15))

        outer_entry = tk.Entry(outer_frame, textvariable=outer_var,
                               validate='key', validatecommand=vcmd_float,
                               font=('Arial', 10), width=10)
        outer_entry.pack(side='left')

        tk.Label(outer_frame, text="(deve ser > raio interno)",
                 font=('Arial', 9), fg='gray').pack(side='left', padx=(10, 0))

        # 5. Fator de zoom
        tk.Label(main_frame, text="Fator de zoom:",
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 5))

        zoom_frame = tk.Frame(main_frame)
        zoom_frame.pack(fill='x', pady=(0, 20))

        zoom_entry = tk.Entry(zoom_frame, textvariable=zoom_var,
                              validate='key', validatecommand=vcmd_float,
                              font=('Arial', 10), width=10)
        zoom_entry.pack(side='left')

        tk.Label(zoom_frame, text="(1.1 a 10.0, onde 2.0 = 2x zoom)",
                 font=('Arial', 9), fg='gray').pack(side='left', padx=(10, 0))

        # Função para validar e aplicar configurações
        def apply_config():
            try:
                # Obtém valores
                new_n_points = n_points_var.get()
                new_aperture = aperture_var.get()
                new_inner = inner_var.get()
                new_outer = outer_var.get()
                new_zoom = zoom_var.get()

                # Validações
                errors = []

                if new_n_points < 3 or new_n_points > 50:
                    errors.append("Número de pontos deve estar entre 3 e 50")

                if new_aperture < 1.0 or new_aperture > 20.0:
                    errors.append("Raio da abertura deve estar entre 1.0 e 20.0")

                if new_inner <= new_aperture:
                    errors.append("Raio interno deve ser maior que o raio da abertura")

                if new_outer <= new_inner:
                    errors.append("Raio externo deve ser maior que o raio interno")

                if new_zoom < 1.1 or new_zoom > 10.0:
                    errors.append("Fator de zoom deve estar entre 1.1 e 10.0")

                # Se há erros, mostra mensagem
                if errors:
                    error_msg = "Erros encontrados:\n\n" + "\n".join(f"• {error}" for error in errors)
                    messagebox.showerror("Erro de Validação", error_msg, parent=config_window)
                    return

                # Aplica as configurações
                self.n_points = new_n_points
                self.aperture_radius = new_aperture
                self.annulus_inner = new_inner
                self.annulus_outer = new_outer
                self.zoom_factor = new_zoom

                # Mensagem de sucesso
                messagebox.showinfo("Sucesso", "Configurações aplicadas com sucesso!",
                                    parent=config_window)

                # Fecha a janela
                config_window.destroy()

                # Se já existe um streak confirmado, atualiza a visualização
                if hasattr(self, 'analysis_points') and self.analysis_points_plot:
                    self.show_analysis_points()

                print("Configurações atualizadas:")
                print(f"  Número de pontos: {self.n_points}")
                print(f"  Raio da abertura: {self.aperture_radius}")
                print(f"  Raio interno: {self.annulus_inner}")
                print(f"  Raio externo: {self.annulus_outer}")
                print(f"  Fator de zoom: {self.zoom_factor}")

            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao aplicar configurações: {str(e)}",
                                     parent=config_window)

        # Função para resetar valores padrão
        def reset_defaults():
            n_points_var.set(10)
            aperture_var.set(5.0)
            inner_var.set(8.0)
            outer_var.set(12.0)
            zoom_var.set(2.0)

        # Frame para botões
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        # Botão para valores padrão
        default_btn = tk.Button(button_frame, text="Valores Padrão",
                                command=reset_defaults,
                                font=('Arial', 10), width=12)
        default_btn.pack(side='left')

        # Botão cancelar
        cancel_btn = tk.Button(button_frame, text="Cancelar",
                               command=config_window.destroy,
                               font=('Arial', 10), width=12)
        cancel_btn.pack(side='right', padx=(5, 0))

        # Botão aplicar
        apply_btn = tk.Button(button_frame, text="Aplicar",
                              command=apply_config,
                              font=('Arial', 10, 'bold'), width=12,
                              bg='lightgreen', activebackground='green')
        apply_btn.pack(side='right', padx=(5, 0))

        # Foco inicial no primeiro campo
        points_entry.focus_set()

        # Aguarda o fechamento da janela
        config_window.wait_window()

    if 'dados_temporais_satelite' not in globals():
        dados_temporais_satelite = []
        tempo_base_satelite = None


    def extrair_tempo_fits(self, fits_file):
        """Extrai timestamp e tempo de exposição do arquivo FITS"""
        try:
            with fits.open(fits_file) as hdul:
                header = hdul[0].header

                # Extrair timestamp
                timestamp = header.get('DATE-OBS', None)
                if timestamp:
                    timestamp = pd.to_datetime(timestamp)

                # Extrair tempo de exposição
                exp_time = header.get('EXPTIME', 1.0)  # default 1 segundo

                return timestamp, exp_time
        except:
            return None, 1.0


    def adicionar_dados_temporais(self,fits_file, positions, fluxes, flux_errs, mags, mag_errs,
                                  snr_vals, fwhm_vals, local_contrast, saturated_flags,
                                  out_z, out_iqr, bkg_vals=None):
        """Adiciona dados da análise atual à estrutura temporal"""
        global dados_temporais_satelite, tempo_base_satelite

        fits_file = self.fits_files[self.current_file_index]

        timestamp, exp_time = self.extrair_tempo_fits(fits_file)

        # Define o tempo base na primeira execução
        if self.tempo_base_satelite is None and timestamp is not None:
            self.tempo_base_satelite = timestamp

        # Calcular tempo relativo
        if timestamp is not None and self.tempo_base_satelite is not None:
            self.tempo_relativo = (timestamp - self.tempo_base_satelite).total_seconds()
        else:
            # Fallback: usar índice se não houver timestamp
            self.tempo_relativo = len(self.dados_temporais_satelite) * 30.0  # assume 30s entre exposições

        # Armazenar dados
        self.dados_temporais_satelite.append({
            'timestamp': timestamp,
            'tempo_relativo': self.tempo_relativo,
            'exp_time': exp_time,
            'arquivo': fits_file,
            'positions': positions.copy(),
            'fluxes': fluxes.copy(),
            'flux_errs': flux_errs.copy(),
            'mags': mags.copy(),
            'mag_errs': mag_errs.copy(),
            'snr_vals': snr_vals.copy(),
            'fwhm_vals': fwhm_vals.copy(),
            'local_contrast': local_contrast.copy(),
            'saturated_flags': saturated_flags.copy(),
            'out_z': out_z.copy(),
            'out_iqr': out_iqr.copy(),
            'bkg_vals': bkg_vals.copy() if bkg_vals is not None else None
        })


    def criar_eixo_temporal_continuo(self, event=None):
        """Cria eixo temporal contínuo e ordena dados"""
        global dados_temporais_satelite

        if not self.dados_temporais_satelite:
            return None, None

        # Ordenar por timestamp (ou tempo relativo se timestamp não disponível)
        dados_ordenados = sorted(self.dados_temporais_satelite,
                                 key=lambda x: x['timestamp'] if x['timestamp'] is not None else x['tempo_relativo'])

        # Criar eixo temporal contínuo
        eixo_temporal = []
        tempo_acumulado = 0

        for i, dados in enumerate(dados_ordenados):
            exp_time = dados['exp_time']
            n_pontos = len(dados['positions'])

            # Criar tempo para cada ponto dentro da exposição
            tempo_pontos = np.linspace(tempo_acumulado, tempo_acumulado + exp_time, n_pontos)
            eixo_temporal.append(tempo_pontos)

            # Atualizar tempo acumulado
            tempo_acumulado += exp_time

            # Adicionar gap entre exposições (fica muito ruim)
            #if i < len(dados_ordenados) - 1:
            #    gap = 0.5  # 0,5 segundo de gap
            #    tempo_acumulado += gap

        return eixo_temporal, dados_ordenados

    def plot_analise_temporal_continua(self,fim_pasta=False):
        """Plota análise fotométrica com tempo contínuo"""
        global dados_temporais_satelite

        resultado = self.criar_eixo_temporal_continuo()
        if resultado[0] is None:
            print("Nenhum dado temporal disponível ainda.")
            return

        eixo_temporal, dados_ordenados = resultado

        # Criar figura expandida
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        # Cores para diferentes exposições
        cores = plt.cm.tab10(np.linspace(0, 1, len(dados_ordenados)))

        # Título (estava ocupando muito espaço)
        #titulo = f'Análise Fotométrica Temporal Contínua ({len(dados_ordenados)} exposições)'
        #if fim_pasta:
        #    titulo += ' - COMPLETA'
        #fig.suptitle(titulo, fontsize=16)

        # Título
        titulo = f'                     '
        # if fim_pasta:
        #    titulo += ' - COMPLETA'
        fig.suptitle(titulo, fontsize=16)


        # Plot de cada análise:

        # 1. Curva de luz temporal contínua
        ax1 = axes[0, 0]
        for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
            valid_points = ~dados['saturated_flags'] & (dados['fluxes'] > 0)

            if np.sum(valid_points) > 0:
                ax1.plot(tempo_pontos[valid_points], dados['fluxes'][valid_points],
                         'o-', color=cores[i], alpha=0.7, markersize=3,
                         label=f'Exp {i + 1}' if i < 5 else '')

            # Marcar outliers
            if len(dados['out_z']) > 0:
                ax1.scatter(tempo_pontos[dados['out_z']], dados['fluxes'][dados['out_z']],
                            c='red', s=50, marker='x', alpha=0.8)

        ax1.set_xlabel('Tempo total (segundos)')
        ax1.set_ylabel('Flux (ADU/s)')
        ax1.set_title('Curva de Luz')
        ax1.grid(True, alpha=0.3)
        if len(dados_ordenados) <= 5:
            ax1.legend()

        # 2. Magnitudes temporais
        ax2 = axes[0, 1]
        for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
            valid_mags = ~np.isnan(dados['mags']) & ~dados['saturated_flags']

            if np.sum(valid_mags) > 0:
                ax2.plot(tempo_pontos[valid_mags], dados['mags'][valid_mags],
                         'o-', color=cores[i], alpha=0.7, markersize=3)

        ax2.set_xlabel('Tempo total (segundos)')
        ax2.set_ylabel('Magnitude instrumental')
        ax2.set_title('Curva de Magnitude')
        ax2.grid(True, alpha=0.3)
        try:
            ax2.invert_yaxis()
        except:
            pass

        # 3. Signal-to-Noise temporal
        ax3 = axes[0, 2]
        for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
            ax3.plot(tempo_pontos, dados['snr_vals'], 'o-', color=cores[i], alpha=0.7, markersize=3)

        ax3.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='S/N = 5')
        ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='S/N = 10')
        ax3.set_xlabel('Tempo total (segundos)')
        ax3.set_ylabel('Signal-to-Noise Ratio')
        ax3.set_title('S/N pelo Tempo')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. Background temporal (estava dando problema por algum motivo :/)
        #ax4 = axes[1, 0]
        #for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
        #    if dados['bkg_vals'] is not None:
        #        ax4.plot(tempo_pontos, dados['bkg_vals'], 'o-', color=cores[i], alpha=0.7, markersize=3)

        #ax4.set_xlabel('Tempo total (segundos)')
        #ax4.set_ylabel('Background (ADU)')
        #ax4.set_title('Variação do Background pelo Tempo')
        #ax4.grid(True, alpha=0.3)

        # 5. Estatísticas por exposição
        ax5 = axes[1, 1]
        flux_medios = []
        tempos_centrais = []

        for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
            valid_points = ~dados['saturated_flags'] & (dados['fluxes'] > 0)
            if np.sum(valid_points) > 0:
                flux_medios.append(np.mean(dados['fluxes'][valid_points]))
                tempos_centrais.append(np.mean(tempo_pontos))

        if flux_medios:
            ax5.plot(tempos_centrais, flux_medios, 'ro-', markersize=8)
            ax5.set_xlabel('Tempo total (segundos)')
            ax5.set_ylabel('Flux médio por exposição')
            ax5.set_title('Evolução do Flux Médio')
            ax5.grid(True, alpha=0.3)

        # 6. FWHM temporal
        ax6 = axes[1, 2]
        for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
            valid_fwhm = ~np.isnan(dados['fwhm_vals'])
            if np.sum(valid_fwhm) > 0:
                ax6.plot(tempo_pontos[valid_fwhm], dados['fwhm_vals'][valid_fwhm],
                         'o-', color=cores[i], alpha=0.7, markersize=3)

        ax6.set_xlabel('Tempo total (segundos)')
        ax6.set_ylabel('FWHM (pixels)')
        ax6.set_title('FWHM')
        ax6.grid(True, alpha=0.3)

        # 7. Contraste temporal
        ax7 = axes[2, 0]
        for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
            valid_contrast = ~np.isnan(dados['local_contrast'])
            if np.sum(valid_contrast) > 0:
                ax7.plot(tempo_pontos[valid_contrast], dados['local_contrast'][valid_contrast],
                         'o-', color=cores[i], alpha=0.7, markersize=3)

        ax7.set_xlabel('Tempo total (segundos)')
        ax7.set_ylabel('Contraste Local')
        ax7.set_title('Contraste')
        ax7.grid(True, alpha=0.3)

        # 8. Contagem de outliers por tempo
        ax8 = axes[2, 1]
        outliers_z = []
        outliers_iqr = []
        tempos_outliers = []

        for i, (tempo_pontos, dados) in enumerate(zip(eixo_temporal, dados_ordenados)):
            outliers_z.append(len(dados['out_z']))
            outliers_iqr.append(len(dados['out_iqr']))
            tempos_outliers.append(np.mean(tempo_pontos))

        if tempos_outliers and len(tempos_outliers) == len(outliers_z) == len(outliers_iqr):
            largura_barra = min(2.0, (max(tempos_outliers) - min(tempos_outliers)) / len(tempos_outliers) * 0.8) if len(
                tempos_outliers) > 1 else 2.0
            ax8.bar([t - largura_barra / 2 for t in tempos_outliers], outliers_z, width=largura_barra, alpha=0.7,
                    label='Outliers Z-score', color='red')
            ax8.bar([t + largura_barra / 2 for t in tempos_outliers], outliers_iqr, width=largura_barra, alpha=0.7,
                    label='Outliers IQR', color='orange')

        ax8.set_xlabel('Tempo total (segundos)')
        ax8.set_ylabel('Número de outliers')
        ax8.set_title('Outliers por Exposição')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Resumo
        ax9 = axes[2, 2]
        # Criar um resumo textual
        ax9.axis('off')

        if eixo_temporal:
            tempo_total = max([max(t) for t in eixo_temporal])
            ultimo_arquivo = os.path.basename(dados_ordenados[-1]['arquivo'])
            pontos_validos_ultimo = np.sum(~dados_ordenados[-1]['saturated_flags'])
            total_pontos_ultimo = len(dados_ordenados[-1]['positions'])
            flux_medio_ultimo = np.mean(dados_ordenados[-1]['fluxes'])
            snr_medio_ultimo = np.mean(dados_ordenados[-1]['snr_vals'])
        else:
            tempo_total = 0
            ultimo_arquivo = "N/A"
            pontos_validos_ultimo = 0
            total_pontos_ultimo = 0
            flux_medio_ultimo = 0
            snr_medio_ultimo = 0




        #texto_resumo = f"""RESUMO

    #Exposições processadas: {len(dados_ordenados)}
    #Tempo total: {max([max(t) for t in eixo_temporal]):.1f} segundos

    #Última exposição:
    #- Arquivo: {os.path.basename(dados_ordenados[-1]['arquivo'])}
    #- Pontos válidos: {np.sum(~dados_ordenados[-1]['saturated_flags'])}/{len(dados_ordenados[-1]['positions'])}
    #- Flux médio: {np.mean(dados_ordenados[-1]['fluxes']):.2f} ADU/s
    #- S/N médio: {np.mean(dados_ordenados[-1]['snr_vals']):.1f}

    #Status: {'COMPLETA' if fim_pasta else 'Em processamento...'}"""

        #ax9.text(0.1, 0.9, texto_resumo, transform=ax9.transAxes, fontsize=11,
                 #verticalalignment='top', fontfamily='monospace',
                 #bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        plt.tight_layout()
        plt.show()

        # Estatísticas resumidas
        print("\n" + "=" * 60)
        print("RESUMO DA ANÁLISE TEMPORAL CONTÍNUA")
        print("=" * 60)
        print(f"Total de exposições: {len(dados_ordenados)}")
        print(f"Tempo total de observação: {max([max(t) for t in eixo_temporal]):.1f} segundos")

        total_pontos = sum(len(d['positions']) for d in dados_ordenados)
        total_saturados = sum(np.sum(d['saturated_flags']) for d in dados_ordenados)
        total_outliers_z = sum(len(d['out_z']) for d in dados_ordenados)
        total_outliers_iqr = sum(len(d['out_iqr']) for d in dados_ordenados)

        print(f"Total de pontos analisados: {total_pontos}")
        print(f"Total de pontos saturados: {total_saturados}")
        print(f"Total de outliers Z-score: {total_outliers_z}")
        print(f"Total de outliers IQR: {total_outliers_iqr}")

        if fim_pasta:
            print("STATUS: Análise temporal COMPLETA")
        else:
            print("STATUS: Análise temporal em andamento...")

        print("=" * 60)

    def complete_analysis(self, event=None):
        """Analisa todas as imagens automaticamente"""

        while (self.current_file_index < len(self.fits_files)):
            self.find_streak()
            self.confirm_streak()
            self.analyze_photometry(fast=False)  # Ativa o sistema que converte a linha principal em tempo
            self.next_image()

    def detect_streak_endpoints(self, event=None) -> Tuple[List[float], List[float]]:
        """
        Detecta os endpoints principais de streaks após filtragem e merge.
        Prioriza ângulos para "costurar" streaks fragmentados em um único streak.
        Remove falsos positivos e unifica streaks cortados.

        Args:
            event: Evento (não usado, compatibilidade)

        Returns:
            Tuple com [x1, y1], [x2, y2] dos endpoints do streak principal
        """

        from detect_streak import StreakCoordinates

        # Utiliza a imagem que está sendo exibida
        current_file = self.fits_files[self.current_file_index]
        streaks, execution_time = detect_streaks_in_fits(
            current_file,
            shape_cut=0.3,
            radius_dev_cut=0.4
        )

        # print(streaks) # Apenas para debugar

        if not streaks:
            messagebox.showwarning("Aviso", "Nenhum Streak Encontrado!")
            return

        # Parâmetros de filtragem focado em ângulos
        min_length = 3  # Comprimento mínimo (para não perder fragmentos)
        max_distance = 300  # Distância máxima aumentada para costura
        angle_tolerance = 2  # Tolerância de ângulo ligeiramente aumentada

        # Filtrar por comprimento mínimo (bem permissivo)
        valid_streaks = [s for s in streaks if s.length >= min_length]

        if not valid_streaks:
            messagebox.showwarning("Aviso", "Nenhum Streak Dentro das especificações!")
            return

        def get_angle(streak):
            """Calcula o ângulo do streak em graus (0-180)"""
            dx = streak.x_end - streak.x_start
            dy = streak.y_end - streak.y_start
            angle = np.arctan2(dy, dx) * 180 / np.pi
            return angle + 180 if angle < 0 else angle


        # Se só tem um streak, retorna seus endpoints
        if len(valid_streaks) == 1:
            streak = valid_streaks[0]
            return [streak.x_start, streak.y_start], [streak.x_end, streak.y_end]

        # Agrupar streaks por ângulo similar
        streak_groups = []
        used_indices = set()

        for i, streak1 in enumerate(valid_streaks):
            if i in used_indices:
                continue

            # Inicia novo grupo com o streak atual
            current_group = [streak1]
            current_indices = {i}
            angle1 = get_angle(streak1)

            # Procura todos os streaks com ângulo similar
            for j, streak2 in enumerate(valid_streaks):
                if j == i or j in used_indices:
                    continue

                angle2 = get_angle(streak2)
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))

                # Se o ângulo está dentro da tolerância, adicionar ao grupo
                if angle_diff <= angle_tolerance:
                    current_group.append(streak2)
                    current_indices.add(j)

            # Verificar se os streaks do grupo podem ser conectados espacialmente
            final_group = [current_group[0]]  # Sempre incluir o streak base
            final_indices = {i}

            for streak in current_group[1:]:
                # Verificar se este streak pode ser conectado com algum do grupo final
                can_connect = False
                for existing_streak in final_group:
                    # Calcular distância mínima entre endpoints
                    endpoints1 = [(existing_streak.x_start, existing_streak.y_start),
                                  (existing_streak.x_end, existing_streak.y_end)]
                    endpoints2 = [(streak.x_start, streak.y_start),
                                  (streak.x_end, streak.y_end)]

                    min_dist = min(np.sqrt((e1[0] - e2[0]) ** 2 + (e1[1] - e2[1]) ** 2)
                                   for e1 in endpoints1 for e2 in endpoints2)

                    if min_dist <= max_distance:
                        can_connect = True
                        break

                if can_connect:
                    final_group.append(streak)
                    final_indices.add(valid_streaks.index(streak))

            if len(final_group) > 0:
                streak_groups.append(final_group)
                used_indices.update(final_indices)

        # Se não conseguiu agrupar nenhum streak, usar todos individualmente
        if not streak_groups:
            streak_groups = [[streak] for streak in valid_streaks]

        # Encontrar o grupo com maior comprimento total
        best_group = None
        max_total_length = 0

        for group in streak_groups:
            # Coletar todas as coordenadas do grupo
            all_coords = []
            for streak in group:
                all_coords.extend(zip(streak.x_coords, streak.y_coords))

            # Calcular comprimento total como distância entre pontos mais extremos
            if len(all_coords) >= 2:
                max_distance = 0
                for i, p1 in enumerate(all_coords):
                    for j, p2 in enumerate(all_coords):
                        if i >= j:
                            continue
                        dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                        if dist > max_distance:
                            max_distance = dist

                if max_distance > max_total_length:
                    max_total_length = max_distance
                    best_group = all_coords

        # Se encontrou um grupo válido, retornar os endpoints mais extremos
        if best_group and len(best_group) >= 2:
            # Encontrar os dois pontos mais distantes (endpoints do streak costurado)
            max_distance = 0
            best_endpoints = None

            for i, p1 in enumerate(best_group):
                for j, p2 in enumerate(best_group):
                    if i >= j:
                        continue
                    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    if dist > max_distance:
                        max_distance = dist
                        best_endpoints = (p1, p2)

            if best_endpoints:
                return list(best_endpoints[0]), list(best_endpoints[1])

        # Fallback: retornar o streak individual mais longo
        longest_streak = max(valid_streaks, key=lambda s: s.length)
        return [longest_streak.x_start, longest_streak.y_start], [longest_streak.x_end, longest_streak.y_end]

    def find_streak(self, event=None):

        p1, p2 = self.detect_streak_endpoints(self.current_data)
        self.streak_points = [p1, p2]

        # desenha pontos e linha
        for idx, (x, y) in enumerate(self.streak_points, start=1):
            self.ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white')
            self.ax.annotate(f'P{idx}', (x, y),
                             xytext=(5, 5), textcoords='offset points',
                             color='red', fontweight='bold')
        x1, y1 = p1;
        x2, y2 = p2
        self.streak_line = self.ax.plot([x1, x2], [y1, y2],
                                        'b-', linewidth=3, alpha=0.7)[0]

        # atualiza título e confirma automaticamente
        self.ax.set_title("Streak detectado automaticamente")
        plt.draw()

    def save_scientific_results(self, results_dict):
        """Salva resultados científicos mas falta implementar direito"""
        print("Resultados científicos salvos:", results_dict.keys())

    def plot_results(self, positions, fluxes, flux_errs, snr_vals, mags, mag_errs,
                                   fwhm_vals, local_contrast, out_z, out_iqr, saturated_flags, fast):

        if fast:
            """Plot expandido dos resultados fotométricos"""
            # Cria figura com subplots (3x3 para incluir mais análises)
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))

            # Título principal
            filename = os.path.basename(self.fits_files[self.current_file_index]) if hasattr(self,
                                                                                          'fits_files') else 'Análise Fotométrica'

            # Retirei o título porque estava ocupando o espaço das outras coisas
            fig.suptitle(f'            ', fontsize=16)

            # Máscaras para diferentes tipos de pontos
            valid_points = ~saturated_flags & (fluxes > 0)
            valid_mags = ~np.isnan(mags) & valid_points
            valid_fwhm = ~np.isnan(fwhm_vals)
            valid_contrast = ~np.isnan(local_contrast)

            # 1. Curva de luz (flux vs posição)
            ax1 = axes[0, 0]
            ax1.errorbar(positions[valid_points], fluxes[valid_points],
                         yerr=flux_errs[valid_points], fmt='bo-', capsize=3, label='Válidos')
            if len(out_z) > 0:
                ax1.scatter(positions[out_z], fluxes[out_z], c='red', s=100,
                            marker='x', label='Outliers Z-score')
            if len(out_iqr) > 0:
                ax1.scatter(positions[out_iqr], fluxes[out_iqr], c='orange', s=80,
                        marker='s', label='Outliers IQR')
            if np.sum(saturated_flags) > 0:
                ax1.scatter(positions[saturated_flags], fluxes[saturated_flags],
                        c='purple', s=100, marker='^', label='Saturados')
            ax1.set_xlabel('Posição no streak')
            ax1.set_ylabel('Flux (ADU/s)')
            ax1.set_title('Curva de Luz')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 2. Magnitudes
            ax2 = axes[0, 1]
            if np.sum(valid_mags) > 0:
                ax2.errorbar(positions[valid_mags], mags[valid_mags],
                         yerr=mag_errs[valid_mags], fmt='go-', capsize=3)
                ax2.invert_yaxis()  # Magnitudes menores = mais brilhante
            ax2.set_xlabel('Posição no streak')
            ax2.set_ylabel('Magnitude instrumental')
            ax2.set_title('Curva de Magnitude')
            ax2.grid(True, alpha=0.3)

            # 3. Signal-to-Noise Ratio
            ax3 = axes[0, 2]
            ax3.plot(positions, snr_vals, 'mo-')
            ax3.axhline(y=5, color='r', linestyle='--', label='S/N = 5')
            ax3.axhline(y=10, color='orange', linestyle='--', label='S/N = 10')
            ax3.set_xlabel('Posição no streak')
            ax3.set_ylabel('Signal-to-Noise Ratio')
            ax3.set_title('Relação Sinal/Ruído')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # 4. Background (usando bkg_vals se disponível)
            ax4 = axes[1, 0]
            if hasattr(self, 'bkg_vals') and len(self.bkg_vals) == len(positions):
                ax4.plot(positions, self.bkg_vals, 'co-')
            else:
                # Fallback: estimar background dos dados
                backgrounds = []
                for i, (x, y) in enumerate(self.analysis_points):
                    an = CircularAnnulus((x, y), r_in=self.annulus_inner, r_out=self.annulus_outer)
                    bt = aperture_photometry(self.current_data, an)
                    backgrounds.append(bt['aperture_sum'][0] / an.area)
                ax4.plot(positions, backgrounds, 'co-')
            ax4.set_xlabel('Posição no streak')
            ax4.set_ylabel('Background (ADU)')
            ax4.set_title('Variação do Background')
            ax4.grid(True, alpha=0.3)

            # 5. Histograma de flux
            ax5 = axes[1, 1]
            valid_flux = fluxes[valid_points]
            if len(valid_flux) > 0:
                ax5.hist(valid_flux, bins=min(10, len(valid_flux) // 2), alpha=0.7, edgecolor='black')
                ax5.axvline(np.mean(valid_flux), color='red', linestyle='--', label='Média')
                ax5.axvline(np.median(valid_flux), color='green', linestyle='--', label='Mediana')
            ax5.set_xlabel('Flux (ADU/s)')
            ax5.set_ylabel('Frequência')
            ax5.set_title('Distribuição de Flux')
            ax5.legend()

            # 6. Erro vs Flux
            ax6 = axes[1, 2]
            ax6.plot(fluxes[valid_points], flux_errs[valid_points], 'ko')
            ax6.set_xlabel('Flux (ADU/s)')
            ax6.set_ylabel('Erro do Flux (ADU/s)')
            ax6.set_title('Erro vs Flux')
            ax6.grid(True, alpha=0.3)

            # 7. FWHM vs Posição
            ax7 = axes[2, 0]
            if np.sum(valid_fwhm) > 0:
                ax7.plot(positions[valid_fwhm], fwhm_vals[valid_fwhm], 'ro-')
                ax7.axhline(y=np.nanmean(fwhm_vals), color='blue', linestyle='--',
                        label=f'Média: {np.nanmean(fwhm_vals):.2f} px')
            ax7.set_xlabel('Posição no streak')
            ax7.set_ylabel('FWHM (pixels)')
            ax7.set_title('Variação do FWHM')
            ax7.grid(True, alpha=0.3)
            ax7.legend()

            # 8. Contraste Local
            ax8 = axes[2, 1]
            if np.sum(valid_contrast) > 0:
                ax8.plot(positions[valid_contrast], local_contrast[valid_contrast], 'go-')
                ax8.axhline(y=np.nanmean(local_contrast), color='red', linestyle='--',
                        label=f'Média: {np.nanmean(local_contrast):.1f}')
            ax8.set_xlabel('Posição no streak')
            ax8.set_ylabel('Contraste (Sinal/Background)')
            ax8.set_title('Contraste Local')
            ax8.grid(True, alpha=0.3)
            ax8.legend()

        # 9. Análise de Outliers
            ax9 = axes[2, 2]
            ax9.plot(positions, fluxes, 'bo-', alpha=0.5, label='Todos os pontos')
            if len(out_z) > 0:
                ax9.scatter(positions[out_z], fluxes[out_z], c='red', s=100,
                            marker='x', label=f'Outliers Z-score ({len(out_z)})')
            if len(out_iqr) > 0:
                ax9.scatter(positions[out_iqr], fluxes[out_iqr], c='orange', s=80,
                        marker='s', label=f'Outliers IQR ({len(out_iqr)})')

            # Adiciona linha de regressão se disponível
            if hasattr(self, 'slope') and hasattr(self, 'intercept'):
                ax9.plot(positions, self.intercept + self.slope * positions,
                     'r--', label='Regressão linear')

            ax9.set_xlabel('Posição no streak')
            ax9.set_ylabel('Flux (ADU/s)')
            ax9.set_title('Detecção de Outliers')
            ax9.grid(True, alpha=0.3)
            ax9.legend()

            plt.tight_layout()
            plt.show()

            # Imprime estatísticas resumidas
            print("\n")
            print("RESUMO DA ANÁLISE FOTOMÉTRICA")
            print("=")
            print(f"Pontos válidos: {np.sum(valid_points)}/{len(positions)}")
            print(f"Pontos saturados: {np.sum(saturated_flags)}")
            print(f"Outliers Z-score: {len(out_z)}")
            print(f"Outliers IQR: {len(out_iqr)}")
            if np.sum(valid_points) > 0:
                print(f"Flux médio: {np.mean(fluxes[valid_points]):.2f} ± {np.std(fluxes[valid_points]):.2f} ADU/s")
                print(f"SNR médio: {np.mean(snr_vals[valid_points]):.2f}")
            if np.sum(valid_mags) > 0:
                print(f"Magnitude média: {np.mean(mags[valid_mags]):.2f} ± {np.std(mags[valid_mags]):.2f}")
            if np.sum(valid_fwhm) > 0:
                print(f"FWHM médio: {np.nanmean(fwhm_vals):.2f} pixels")
            if np.sum(valid_contrast) > 0:
                print(f"Contraste médio: {np.nanmean(local_contrast):.1f}")
            print("=" * 50)

        else:

            def executar_analise_temporal(fits_file, positions, fluxes, flux_errs, mags, mag_errs,
                                          snr_vals, fwhm_vals, local_contrast, saturated_flags,
                                          out_z, out_iqr, bkg_vals=None, fim_pasta=False):
                """
                Função principal para executar no seu if

                Parâmetros:
                - fits_file: caminho do arquivo FITS atual
                - positions, fluxes, etc.: seus arrays de dados
                - fim_pasta: True quando é o último arquivo da pasta
                """

                # Adicionar dados atuais
                self.adicionar_dados_temporais(fits_file, positions, fluxes, flux_errs, mags, mag_errs,
                                          snr_vals, fwhm_vals, local_contrast, saturated_flags,
                                          out_z, out_iqr, bkg_vals)

                # Plotar análise temporal
                self.plot_analise_temporal_continua(fim_pasta)

                # Retornar dados temporais se necessário
                return self.dados_temporais_satelite

            # Verifica se é o fim da pasta

            print("############################## - Dados adquiridos")
            print("Imagem: ", self.current_file_index, " de ", len(self.fits_files) - 1)
            print("********")


            fim_pasta = False
            index = self.current_file_index
            max = len(self.fits_files) - 1

            if index == max:
                print("############################## - Fim da pasta declarado")
                fim_pasta = True

            current_file = self.fits_files[self.current_file_index]


            executar_analise_temporal(current_file, positions, fluxes, flux_errs, mags, mag_errs,
                                      snr_vals, fwhm_vals, local_contrast, saturated_flags,
                                      out_z, out_iqr, bkg_vals=None, fim_pasta=fim_pasta)


    def generate_quality_report(self, valid, saturated_flags, cv, mean_fwhm, mean_contrast, dominant_period):
        """Gera relatório de qualidade"""
        # Foi removido da implementação
        report = f"Saturados: {np.sum(saturated_flags)}, CV: {cv:.3f}"
        return report



if __name__ == "__main__":
    # Cria e executa o programa
    app = SatellitePhotometry()
    app.run()