import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface BeardTemplate {
  id: string
  name: string
  createdAt: string
  beardVertexIndices: number[]
  boundaryVertexIndices: number[]
  calibrationViews: string[]
  templateData?: {
    template_base64: string
    contour: number[][]
    frame_count: number
    canonical_size: number
  }
}

interface BeardState {
  // Templates
  savedTemplates: BeardTemplate[]
  activeTemplateId: string | null
  
  // Live tracking
  currentLandmarks: number[][] | null
  
  // Actions
  addTemplate: (template: BeardTemplate) => void
  removeTemplate: (id: string) => void
  setActiveTemplate: (id: string | null) => void
  setCurrentLandmarks: (landmarks: number[][] | null) => void
}

export const useBeardStore = create<BeardState>()(
  persist(
    (set) => ({
      savedTemplates: [],
      activeTemplateId: null,
      currentLandmarks: null,
      
      addTemplate: (template) =>
        set((state) => ({
          savedTemplates: [template, ...state.savedTemplates.filter(t => t.id !== template.id)],
          activeTemplateId: template.id,
        })),
      
      removeTemplate: (id) =>
        set((state) => ({
          savedTemplates: state.savedTemplates.filter((t) => t.id !== id),
          activeTemplateId: state.activeTemplateId === id ? null : state.activeTemplateId,
        })),
      
      setActiveTemplate: (id) =>
        set({ activeTemplateId: id }),
      
      setCurrentLandmarks: (landmarks) =>
        set({ currentLandmarks: landmarks }),
    }),
    {
      name: 'beard-storage',
      partialize: (state) => ({
        savedTemplates: state.savedTemplates,
        activeTemplateId: state.activeTemplateId,
      }),
    }
  )
)
