import "dotenv/config";
import { google, drive_v3, docs_v1 } from "googleapis";
import fs from "fs";
import path from "path";

export interface GoogleDocResult {
  success: boolean;
  docId?: string;
  docUrl?: string;
  folderId?: string;
  folderUrl?: string;
  error?: string;
  apiDisabled?: boolean;
  enableApiUrl?: string;
}

export interface ScriptDocInput {
  title: string;
  scriptText: string;           // Tab 1 text (Competitor Transcript)
  supernovaScriptText?: string; // Tab 2 text (Supernova Script)
  hookFramePath?: string;       // Local JPEG path for 1.5s Hook frame
  bodyFramePath?: string;       // Local JPEG path for 30th frame Body frame
  isSameVisualScene?: boolean;  // True when Hook and Body frames are the same visual screen
  folderName?: string;
  existingDocId?: string;       // Optional: if provided, updates this existing Google Doc instead of creating a new file
}

export class GoogleDocsService {
  private drive: drive_v3.Drive | null = null;
  private docs: docs_v1.Docs | null = null;
  private targetFolderName = "Supernova Competitive Transcription";
  private defaultSharedDriveId = "0ANOulSuO9NLjUk9PVA";
  private cachedFolderId: string | null = null;
  private tempUploadedFileIds: string[] = [];

  constructor() {
    this.initAuth();
  }

  private initAuth() {
    try {
      const credPath =
        process.env.GOOGLE_APPLICATION_CREDENTIALS ||
        (fs.existsSync(path.join(process.cwd(), "task-assigner", "credentials.json"))
          ? path.join(process.cwd(), "task-assigner", "credentials.json")
          : null);

      if (credPath && fs.existsSync(credPath)) {
        const auth = new google.auth.GoogleAuth({
          keyFile: credPath,
          scopes: [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets",
          ],
        });
        this.drive = google.drive({ version: "v3", auth });
        this.docs = google.docs({ version: "v1", auth });
        return;
      }

      if (process.env.GOOGLE_SERVICE_ACCOUNT_JSON) {
        const credentials = JSON.parse(process.env.GOOGLE_SERVICE_ACCOUNT_JSON);
        const auth = new google.auth.GoogleAuth({
          credentials,
          scopes: [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets",
          ],
        });
        this.drive = google.drive({ version: "v3", auth });
        this.docs = google.docs({ version: "v1", auth });
        return;
      }
    } catch (err: any) {
      console.warn("[GoogleDocsService] Auth init warning:", err.message);
    }
  }

  public isConfigured(): boolean {
    return Boolean(this.drive);
  }

  /**
   * Finds or returns the Shared Drive / folder ID
   */
  async ensureFolder(folderName = this.targetFolderName): Promise<{ folderId: string; folderUrl: string }> {
    if (!this.drive) {
      throw new Error("Google Drive client is not configured");
    }

    // 0. Use configured folder / Shared Drive ID
    const targetId = process.env.GOOGLE_DRIVE_FOLDER_ID || this.defaultSharedDriveId;
    if (targetId) {
      this.cachedFolderId = targetId;
      return {
        folderId: targetId,
        folderUrl: `https://drive.google.com/drive/folders/${targetId}`,
      };
    }

    if (this.cachedFolderId) {
      return {
        folderId: this.cachedFolderId,
        folderUrl: `https://drive.google.com/drive/folders/${this.cachedFolderId}`,
      };
    }

    return {
      folderId: this.defaultSharedDriveId,
      folderUrl: `https://drive.google.com/drive/folders/${this.defaultSharedDriveId}`,
    };
  }

  /**
   * Creates a new 2-tab Google Doc or updates an existing Google Doc (`existingDocId`)
   * with Tab 1 ("Competitor Transcript"), Tab 2 ("Supernova Script"), small side-by-side
   * visual frame images at the bottom, and clean black/bold formatting.
   */
  async createScriptDoc(params: ScriptDocInput): Promise<GoogleDocResult> {
    if (!this.drive || !this.docs) {
      return {
        success: false,
        error: "Google Drive/Docs client is not configured",
      };
    }

    try {
      // 1. Ensure parent folder exists (Shared Drive 0ANOulSuO9NLjUk9PVA)
      const { folderId, folderUrl } = await this.ensureFolder(params.folderName);

      let docId: string;
      let docUrl: string;
      let primaryTabId = "t.0";

      // 2. Create or update Tab 1 ("Competitor Transcript")
      if (params.existingDocId) {
        docId = params.existingDocId;
        docUrl = `https://docs.google.com/document/d/${docId}/edit?usp=drivesdk`;

        if (params.title) {
          try {
            await this.drive.files.update({
              fileId: docId,
              supportsAllDrives: true,
              requestBody: { name: params.title },
            });
          } catch {
            // Ignore title rename errors on existing docs
          }
        }

        const existingDoc = await this.docs.documents.get({
          documentId: docId,
          includeTabsContent: true,
        });
        const primaryTab = existingDoc.data.tabs?.[0];
        primaryTabId = primaryTab?.tabProperties?.tabId || "t.0";
        const tab1Content =
          primaryTab?.documentTab?.body?.content ||
          existingDoc.data.body?.content ||
          [];
        const tab1EndIndex = tab1Content[tab1Content.length - 1]?.endIndex || 1;

        const tab1Requests: docs_v1.Schema$Request[] = [
          {
            updateDocumentTabProperties: {
              tabProperties: {
                tabId: primaryTabId,
                title: "Competitor Transcript",
              },
              fields: "title",
            },
          },
        ];

        if (tab1EndIndex > 2) {
          tab1Requests.push({
            deleteContentRange: {
              range: {
                startIndex: 1,
                endIndex: tab1EndIndex - 1,
                tabId: primaryTabId,
              },
            },
          });
        }

        if (params.scriptText) {
          tab1Requests.push({
            insertText: {
              location: {
                index: 1,
                tabId: primaryTabId,
              },
              text: params.scriptText,
            },
          });
        }

        await this.docs.documents.batchUpdate({
          documentId: docId,
          requestBody: { requests: tab1Requests },
        });
      } else {
        const createRes = await this.drive.files.create({
          requestBody: {
            name: params.title,
            mimeType: "application/vnd.google-apps.document",
            parents: [folderId],
          },
          media: {
            mimeType: "text/plain",
            body: params.scriptText,
          },
          supportsAllDrives: true,
          fields: "id, name, webViewLink",
        });

        docId = createRes.data.id!;
        docUrl =
          createRes.data.webViewLink ||
          `https://docs.google.com/document/d/${docId}/edit?usp=drivesdk`;

        const createdDoc = await this.docs.documents.get({
          documentId: docId,
          includeTabsContent: true,
        });
        primaryTabId = createdDoc.data.tabs?.[0]?.tabProperties?.tabId || "t.0";

        await this.setDocumentTabTitle(docId, "Competitor Transcript", primaryTabId);
      }

      // 3. Create or update Tab 2 ("Supernova Script") if supernovaScriptText is provided
      let secondTabId: string | undefined;
      if (params.supernovaScriptText) {
        const docWithTabs = await this.docs.documents.get({
          documentId: docId,
          includeTabsContent: true,
        });
        const existingSecondTab = docWithTabs.data.tabs?.[1];

        if (!existingSecondTab) {
          const addTabRes = await this.docs.documents.batchUpdate({
            documentId: docId,
            requestBody: {
              requests: [
                {
                  addDocumentTab: {
                    tabProperties: {
                      title: "Supernova Script",
                    },
                  },
                } as any,
              ],
            },
          });

          secondTabId = (addTabRes.data.replies?.[0] as any)?.addDocumentTab
            ?.tabProperties?.tabId;

          if (!secondTabId) {
            const refreshedDoc = await this.docs.documents.get({
              documentId: docId,
              includeTabsContent: true,
            });
            secondTabId = refreshedDoc.data.tabs?.[1]?.tabProperties?.tabId || undefined;
          }

          if (secondTabId) {
            await this.docs.documents.batchUpdate({
              documentId: docId,
              requestBody: {
                requests: [
                  {
                    insertText: {
                      location: {
                        index: 1,
                        tabId: secondTabId,
                      },
                      text: params.supernovaScriptText,
                    },
                  },
                ],
              },
            });
          }
        } else {
          secondTabId = existingSecondTab.tabProperties?.tabId || undefined;
          if (secondTabId) {
            const tab2Content = existingSecondTab.documentTab?.body?.content || [];
            const tab2EndIndex = tab2Content[tab2Content.length - 1]?.endIndex || 1;
            const tab2Requests: docs_v1.Schema$Request[] = [
              {
                updateDocumentTabProperties: {
                  tabProperties: {
                    tabId: secondTabId,
                    title: "Supernova Script",
                  },
                  fields: "title",
                },
              },
            ];

            if (tab2EndIndex > 2) {
              tab2Requests.push({
                deleteContentRange: {
                  range: {
                    startIndex: 1,
                    endIndex: tab2EndIndex - 1,
                    tabId: secondTabId,
                  },
                },
              });
            }

            tab2Requests.push({
              insertText: {
                location: {
                  index: 1,
                  tabId: secondTabId,
                },
                text: params.supernovaScriptText,
              },
            });

            await this.docs.documents.batchUpdate({
              documentId: docId,
              requestBody: { requests: tab2Requests },
            });
          }
        }
      }

      // 4. Upload temporary visual frame images (if provided) and append at the bottom of each tab
      try {
        let hookUri: string | null = null;
        let bodyUri: string | null = null;
        const isSameScene = Boolean(params.isSameVisualScene);

        if (params.hookFramePath && fs.existsSync(params.hookFramePath)) {
          hookUri = await this.uploadTempFrameImage(
            params.hookFramePath,
            `temp_hook_${Date.now()}.jpg`,
            folderId
          );
        }
        if (!isSameScene && params.bodyFramePath && fs.existsSync(params.bodyFramePath)) {
          bodyUri = await this.uploadTempFrameImage(
            params.bodyFramePath,
            `temp_body_${Date.now()}.jpg`,
            folderId
          );
        }

        if (hookUri || bodyUri) {
          await this.appendVisualFramesToTab(
            docId,
            primaryTabId,
            hookUri,
            bodyUri,
            isSameScene
          );
          if (secondTabId) {
            await this.appendVisualFramesToTab(
              docId,
              secondTabId,
              hookUri,
              bodyUri,
              isSameScene
            );
          }
        }
      } finally {
        await this.cleanupTempFrameImages();
      }

      // 5. Apply clean black text, hyperlinks, and bold formatting to each tab
      await this.formatTabStyles(docId, primaryTabId);
      if (secondTabId) {
        await this.formatTabStyles(docId, secondTabId);
      }

      // 6. Make completely editable by anyone with the link
      try {
        await this.drive.permissions.create({
          fileId: docId,
          supportsAllDrives: true,
          requestBody: {
            role: "writer",
            type: "anyone",
          },
        });
      } catch {
        // Permissions may be inherited automatically from the Shared Drive
      }

      return {
        success: true,
        docId,
        docUrl,
        folderId,
        folderUrl,
      };
    } catch (err: any) {
      console.error("[GoogleDocsService] Failed to create/update script doc:", err.message);
      const isQuotaError = err.message?.includes("storage quota has been exceeded");
      const isApiDisabled =
        err.apiDisabled ||
        err.message?.includes("disabled") ||
        err.message?.includes("has not been used in project") ||
        err.message?.includes("does not have permission");

      const enableUrl =
        "https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=marketing-automation-511412";

      return {
        success: false,
        error: isQuotaError
          ? "Service account storage quota issue in My Drive. Shared Drive recommended."
          : err.message,
        apiDisabled: isApiDisabled,
        enableApiUrl: isApiDisabled ? enableUrl : undefined,
      };
    }
  }

  /**
   * Updates an existing Google Doc by ID (or via ScriptDocInput with existingDocId).
   */
  async updateScriptDoc(
    docIdOrParams: string | (ScriptDocInput & { existingDocId: string }),
    maybeParams?: Omit<ScriptDocInput, "existingDocId" | "title"> & { title?: string }
  ): Promise<GoogleDocResult> {
    if (typeof docIdOrParams === "string") {
      return this.createScriptDoc({
        title: maybeParams?.title || "",
        scriptText: maybeParams?.scriptText || "",
        supernovaScriptText: maybeParams?.supernovaScriptText,
        hookFramePath: maybeParams?.hookFramePath,
        bodyFramePath: maybeParams?.bodyFramePath,
        isSameVisualScene: maybeParams?.isSameVisualScene,
        folderName: maybeParams?.folderName,
        existingDocId: docIdOrParams,
      });
    }
    return this.createScriptDoc(docIdOrParams);
  }

  /**
   * Uploads a local JPEG frame to Drive temporarily, makes it publicly readable for Docs ingestion,
   * and returns the direct public URL.
   */
  async uploadTempFrameImage(
    filePath: string,
    name: string,
    folderId: string
  ): Promise<string | null> {
    if (!this.drive) return null;
    try {
      const res = await this.drive.files.create({
        requestBody: {
          name,
          mimeType: "image/jpeg",
          parents: [folderId],
        },
        media: {
          mimeType: "image/jpeg",
          body: fs.createReadStream(filePath),
        },
        supportsAllDrives: true,
        fields: "id, webContentLink",
      });

      const fileId = res.data.id;
      if (!fileId) return null;

      this.tempUploadedFileIds.push(fileId);

      try {
        await this.drive.permissions.create({
          fileId,
          supportsAllDrives: true,
          requestBody: {
            role: "reader",
            type: "anyone",
          },
        });
      } catch {
        // Shared drive may already allow access
      }

      return `https://drive.google.com/uc?export=view&id=${fileId}`;
    } catch (err: any) {
      console.warn(`[GoogleDocsService] Failed to upload temp frame image (${name}):`, err.message);
      return null;
    }
  }

  /**
   * Deletes temporary frame images from Google Drive after `insertInlineImage` completes.
   */
  private async cleanupTempFrameImages(): Promise<void> {
    if (!this.drive || this.tempUploadedFileIds.length === 0) return;
    const idsToDelete = this.tempUploadedFileIds.splice(0);
    for (const fileId of idsToDelete) {
      try {
        await this.drive.files.update({
          fileId,
          supportsAllDrives: true,
          requestBody: { trashed: true },
        });
      } catch {
        // Ignore if Shared Drive contributor role does not allow trashing/deleting
      }
    }
  }

  /**
   * Appends the visual frame header ("Visual Frame:" for a single/same scene, or
   * "Visual Frames (Left: 1.5s Hook | Right: 30th Frame Body):" for distinct Hook & Body frames)
   * and inserts the thumbnail(s) at the bottom of the specified tab.
   */
  private async appendVisualFramesToTab(
    documentId: string,
    tabId: string,
    hookUri: string | null,
    bodyUri: string | null,
    isSameVisualScene = false
  ): Promise<void> {
    if (!this.docs || (!hookUri && !bodyUri)) return;
    try {
      const doc = await this.docs.documents.get({
        documentId,
        includeTabsContent: true,
      });
      const tab =
        doc.data.tabs?.find((t) => t.tabProperties?.tabId === tabId) ||
        doc.data.tabs?.[0];
      const content = tab?.documentTab?.body?.content || [];
      const lastElem = content[content.length - 1];
      let insertIndex = Math.max(1, (lastElem?.endIndex ?? 2) - 1);

      const singleScene = isSameVisualScene || !(hookUri && bodyUri);
      const headerText = singleScene
        ? "\n\nVisual Frame:\n"
        : "\n\nVisual Frames (Hook & Body):\n";

      const requests: docs_v1.Schema$Request[] = [
        {
          insertText: {
            location: { index: insertIndex, tabId },
            text: headerText,
          },
        },
      ];
      insertIndex += headerText.length;

      const thumbnailSize = {
        width: { magnitude: 90, unit: "PT" },
        height: { magnitude: 160, unit: "PT" },
      };

      if (singleScene) {
        const singleUri = hookUri || bodyUri;
        if (singleUri) {
          requests.push({
            insertInlineImage: {
              location: { index: insertIndex, tabId },
              uri: singleUri,
              objectSize: thumbnailSize,
            },
          });
          insertIndex += 1;
        }
      } else {
        if (hookUri) {
          requests.push({
            insertInlineImage: {
              location: { index: insertIndex, tabId },
              uri: hookUri,
              objectSize: thumbnailSize,
            },
          });
          insertIndex += 1;
        }

        if (hookUri && bodyUri) {
          requests.push({
            insertText: {
              location: { index: insertIndex, tabId },
              text: "  ",
            },
          });
          insertIndex += 2;
        }

        if (bodyUri) {
          requests.push({
            insertInlineImage: {
              location: { index: insertIndex, tabId },
              uri: bodyUri,
              objectSize: thumbnailSize,
            },
          });
          insertIndex += 1;
        }
      }

      await this.docs.documents.batchUpdate({
        documentId,
        requestBody: { requests },
      });
    } catch (err: any) {
      console.warn(
        `[GoogleDocsService] Could not append visual frames to tab ${tabId}:`,
        err.message
      );
    }
  }

  /**
   * Sets the title of a document tab (defaults to primary tab "Competitor Transcript").
   */
  async setDocumentTabTitle(
    documentId: string,
    title = "Competitor Transcript",
    tabId?: string
  ): Promise<void> {
    if (!this.docs) return;
    try {
      let targetTabId = tabId;
      if (!targetTabId) {
        const doc = await this.docs.documents.get({
          documentId,
          includeTabsContent: true,
        });
        targetTabId = doc.data.tabs?.[0]?.tabProperties?.tabId || "t.0";
      }
      await this.docs.documents.batchUpdate({
        documentId,
        requestBody: {
          requests: [
            {
              updateDocumentTabProperties: {
                tabProperties: {
                  tabId: targetTabId,
                  title,
                },
                fields: "title",
              },
            },
          ],
        },
      });
    } catch (err: any) {
      console.warn("[GoogleDocsService] Could not set document tab title:", err.message);
    }
  }

  /**
   * Applies clean black text (no yellow highlighting), clickable hyperlinks,
   * bold section headers, bold character names in the Characters list (before `--`),
   * and bold character names + colons on dialogue lines for a given tab.
   */
  async formatTabStyles(documentId: string, tabId: string): Promise<void> {
    if (!this.docs) return;

    try {
      const doc = await this.docs.documents.get({
        documentId,
        includeTabsContent: true,
      });
      const tab =
        doc.data.tabs?.find((t) => t.tabProperties?.tabId === tabId) ||
        doc.data.tabs?.[0];
      const body =
        tab?.documentTab?.body?.content ||
        (tabId === "t.0" ? doc.data.body?.content : []) ||
        [];

      const requests: docs_v1.Schema$Request[] = [];

      const sectionHeaders = [
        "Visual Description (Stitched Frame — Left: 3s Hook | Right: 30th Frame Body):",
        "Visual Description (Left: 3s Hook | Right: 30th Frame Body):",
        "Visual Description (Stitched Frame — Left: 1.5s Hook | Right: 30th Frame Body):",
        "Visual Description (Left: 1.5s Hook | Right: 30th Frame Body):",
        "Visual Description:",
        "Split Screen & Characters:",
        "Hook visuals:",
        "Hook visuals (3s - Left Side):",
        "Hook visuals (3s):",
        "Hook visuals (1.5s - Left Side):",
        "Hook visuals (1.5s):",
        "Body visuals:",
        "Body visuals (30th frame - Right Side):",
        "Body visuals (30th frame):",
        "Video flow:",
        "Reference link :",
        "Reference link:",
        "Visual Frames (Hook & Body):",
        "Visual Frames (Left: 3s Hook | Right: 30th Frame Body):",
        "Visual Frames (3s Hook & 30th Frame Body):",
        "Visual Frames (Left: 1.5s Hook | Right: 30th Frame Body):",
        "Visual Frames (1.5s Hook & 30th Frame Body):",
        "Visual Frame:",
      ];

      for (const elem of body) {
        if (!elem.paragraph) continue;
        const elements = elem.paragraph.elements || [];
        if (elements.length === 0 || elements[0].startIndex === undefined) continue;

        const paraStartIndex = elements[0].startIndex!;

        // Baseline reset for every text run in this paragraph: clean black text, no highlight, regular weight
        for (const e of elements) {
          if (
            e.textRun?.content &&
            e.startIndex !== undefined &&
            e.endIndex !== undefined &&
            e.endIndex > e.startIndex
          ) {
            requests.push({
              updateTextStyle: {
                range: {
                  startIndex: e.startIndex,
                  endIndex: e.endIndex,
                  tabId,
                },
                textStyle: {
                  bold: false,
                  backgroundColor: {},
                  foregroundColor: {
                    color: {
                      rgbColor: { red: 0, green: 0, blue: 0 },
                    },
                  },
                },
                fields: "bold,backgroundColor,foregroundColor",
              },
            });
          }
        }

        // Reconstruct paragraph string preserving exact UTF-16 index alignment
        const paraText = elements
          .map((e) =>
            e.textRun?.content !== undefined
              ? e.textRun.content
              : "\uFFFC".repeat(Math.max(0, (e.endIndex ?? 0) - (e.startIndex ?? 0)))
          )
          .join("");

        const lineEnd = paraStartIndex + paraText.replace(/\n$/, "").length;

        // 1. Hyperlink URLs
        const urlRegex = /(https?:\/\/[^\s\n\)\<\>\"']+)/g;
        let urlMatch: RegExpExecArray | null;
        while ((urlMatch = urlRegex.exec(paraText)) !== null) {
          const url = urlMatch[0];
          const startIndex = paraStartIndex + urlMatch.index;
          const endIndex = startIndex + url.length;
          if (endIndex > startIndex) {
            requests.push({
              updateTextStyle: {
                range: { startIndex, endIndex, tabId },
                textStyle: {
                  link: { url },
                  foregroundColor: {
                    color: {
                      rgbColor: { red: 0.06, green: 0.45, blue: 0.89 },
                    },
                  },
                  underline: true,
                },
                fields: "link,foregroundColor,underline",
              },
            });
          }
        }

        // 2. Bold Section Headers
        let matchedSectionHeader = false;
        for (const header of sectionHeaders) {
          const idx = paraText.indexOf(header);
          if (idx !== -1) {
            const startIndex = paraStartIndex + idx;
            const endIndex = startIndex + header.length;
            if (endIndex > startIndex) {
              requests.push({
                updateTextStyle: {
                  range: { startIndex, endIndex, tabId },
                  textStyle: { bold: true },
                  fields: "bold",
                },
              });
            }
            if (lineEnd > endIndex) {
              requests.push({
                updateTextStyle: {
                  range: { startIndex: endIndex, endIndex: lineEnd, tabId },
                  textStyle: { bold: false },
                  fields: "bold",
                },
              });
            }
            matchedSectionHeader = true;
          }
        }

        const trimmedPara = paraText.trim();
        if (
          trimmedPara === "Characters" ||
          trimmedPara === "Characters:" ||
          trimmedPara === "Script" ||
          trimmedPara === "Script:"
        ) {
          const idx = paraText.indexOf(trimmedPara);
          const startIndex = paraStartIndex + idx;
          const endIndex = startIndex + trimmedPara.length;
          if (endIndex > startIndex) {
            requests.push({
              updateTextStyle: {
                range: { startIndex, endIndex, tabId },
                textStyle: { bold: true },
                fields: "bold",
              },
            });
          }
          matchedSectionHeader = true;
        }

        // 3. Bold Character Names in the `Characters` List (lines starting with `• ` or `- ` before `--`)
        const bulletMatch = /^([•\-]\s+)([^\n]+?)\s*(--|—|–)(.*)$/s.exec(paraText);
        if (bulletMatch) {
          const prefixLen = bulletMatch[1].length;
          const charName = bulletMatch[2];
          const nameStart = paraStartIndex + prefixLen;
          const nameEnd = nameStart + charName.length;
          if (nameEnd > nameStart) {
            requests.push({
              updateTextStyle: {
                range: { startIndex: nameStart, endIndex: nameEnd, tabId },
                textStyle: { bold: true },
                fields: "bold",
              },
            });
          }
          if (lineEnd > nameEnd) {
            requests.push({
              updateTextStyle: {
                range: { startIndex: nameEnd, endIndex: lineEnd, tabId },
                textStyle: { bold: false },
                fields: "bold",
              },
            });
          }
          continue;
        }

        // 4. Bold Character Names in Dialogue Lines (`/^(?:[A-Z][A-Za-z0-9 ()\/-]{1,40}:)\s/`)
        if (!matchedSectionHeader) {
          const dialogueMatch = /^([A-Z][A-Za-z0-9 ()\/-]{1,40}:)\s/.exec(paraText);
          if (dialogueMatch) {
            const label = dialogueMatch[1];
            const labelStart = paraStartIndex;
            const labelEnd = labelStart + label.length;
            if (labelEnd > labelStart) {
              requests.push({
                updateTextStyle: {
                  range: { startIndex: labelStart, endIndex: labelEnd, tabId },
                  textStyle: { bold: true },
                  fields: "bold",
                },
              });
            }
            if (lineEnd > labelEnd) {
              requests.push({
                updateTextStyle: {
                  range: { startIndex: labelEnd, endIndex: lineEnd, tabId },
                  textStyle: { bold: false },
                  fields: "bold",
                },
              });
            }
          }
        }
      }

      if (requests.length > 0) {
        await this.docs.documents.batchUpdate({
          documentId,
          requestBody: { requests },
        });
      }
    } catch (err: any) {
      console.warn(
        `[GoogleDocsService] Could not format tab styles for tab ${tabId}:`,
        err.message
      );
    }
  }

  /**
   * Backwards-compatible helper: formats the primary tab URLs and styles.
   */
  async hyperlinkDocumentUrls(documentId: string): Promise<void> {
    await this.formatTabStyles(documentId, "t.0");
  }

  /**
   * Backwards-compatible helper: formats the primary tab character labels.
   */
  async formatCharacterLabelsBold(documentId: string): Promise<void> {
    await this.formatTabStyles(documentId, "t.0");
  }
}

export const googleDocsService = new GoogleDocsService();
